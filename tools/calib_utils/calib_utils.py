import copy
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from sklearn.mixture import GaussianMixture
from scipy.optimize import differential_evolution

from pcdet.config import cfg
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils.spconv_utils import spconv
from pcdet.utils import commu_utils


class ProbModel(ABC):
    @abstractmethod
    def fit(self, data):
        pass

    @abstractmethod
    def pred(self, data):
        pass


class GMM(ProbModel):
    def __init__(self, n_components):
        self.model = GaussianMixture(n_components=n_components, max_iter=1000)

    def fit(self, data):
        self.model = self.model.fit(
            data.cpu().numpy() if torch.is_tensor(data) else data)
        return self

    def pred(self, data):
        return self.model.score(data.cpu().numpy() if torch.is_tensor(data) else data) * -1.


class AnchorCalibrator:
    def __init__(self, pretrained_model, dist, logger):
        self.pretrained_model = pretrained_model
        self.dist = dist
        self.logger = logger

    @staticmethod
    def _get_anchors(class_id):
        return np.array(cfg.MODEL.DENSE_HEAD.ANCHOR_GENERATOR_CONFIG[class_id]['anchor_sizes'][0])

    @staticmethod
    def _set_anchors(class_id, anchors):
        # this will set the anchors in the global config
        # make sure to rebuild the model after calling this function
        cfg.MODEL.DENSE_HEAD.ANCHOR_GENERATOR_CONFIG[class_id]['anchor_sizes'][0] = anchors

    @torch.no_grad()
    def get_instance_features_PartA2Net(self, model, data_dict, class_id, threshold, anchors=None):
        # ignore network's size residuals if we overwrite anchors
        data_dict['ignore_size_residuals'] = anchors is not None
        data_dict['return_data_dict'] = True  # for ddp
        pred_dicts, _, data_dict = model(data_dict)

        pooled_features = []
        for batch_index, pred_dict in enumerate(pred_dicts):
            pred_labels = pred_dict['pred_labels']
            pred_scores = pred_dict['pred_scores']
            mask = (pred_labels == (class_id + 1)) & (pred_scores > threshold)

            pred_boxes = pred_dict['pred_boxes'][mask]

            if anchors is not None:
                pred_boxes[:, 3:6] = torch.from_numpy(anchors)

            if pred_boxes.numel() == 0:
                continue

            # create fake batch and use it to pool the features at the predicted locations
            fake_batch = {}
            fake_batch['batch_size'] = 1
            fake_batch['rois'] = pred_boxes.unsqueeze(0)
            mask = data_dict['point_coords'][:, 0] == batch_index
            fake_batch['point_coords'] = data_dict['point_coords'][mask]
            fake_batch['point_coords'][:, 0] = 0
            fake_batch['point_features'] = data_dict['point_features'][mask]
            fake_batch['point_cls_scores'] = data_dict['point_cls_scores'][mask]
            fake_batch['point_part_offset'] = data_dict['point_part_offset'][mask]

            roi_head = model.module.roi_head if commu_utils.get_world_size() > 1 else model.roi_head

            # RoI aware pooling
            pooled_part_features, pooled_rpn_features = roi_head.roiaware_pool(
                fake_batch)
            # (B * N, out_x, out_y, out_z, 4)
            batch_size_rcnn = pooled_part_features.shape[0]

            # transform to sparse tensors
            sparse_shape = np.array(
                pooled_part_features.shape[1:4], dtype=np.int32)
            # (non_empty_num, 4) ==> [bs_idx, x_idx, y_idx, z_idx]
            sparse_idx = pooled_part_features.sum(dim=-1).nonzero()
            if sparse_idx.shape[0] < 3:
                sparse_idx = roi_head.fake_sparse_idx(
                    sparse_idx, batch_size_rcnn)

            part_features = pooled_part_features[sparse_idx[:, 0],
                                                 sparse_idx[:, 1], sparse_idx[:, 2], sparse_idx[:, 3]]
            rpn_features = pooled_rpn_features[sparse_idx[:, 0],
                                               sparse_idx[:, 1], sparse_idx[:, 2], sparse_idx[:, 3]]
            coords = sparse_idx.int().contiguous()
            part_features = spconv.SparseConvTensor(
                part_features, coords, sparse_shape, batch_size_rcnn)
            rpn_features = spconv.SparseConvTensor(
                rpn_features, coords, sparse_shape, batch_size_rcnn)

            # forward rcnn network
            x_part = roi_head.conv_part(part_features)
            x_rpn = roi_head.conv_rpn(rpn_features)

            merged_feature = torch.cat(
                (x_rpn.features, x_part.features), dim=1)  # (N, C)
            shared_feature = spconv.SparseConvTensor(
                merged_feature, coords, sparse_shape, batch_size_rcnn)
            shared_feature = shared_feature.dense().view(batch_size_rcnn, -1, 1)

            for layer_i in range(8):
                shared_feature = roi_head.shared_fc_layer[layer_i](
                    shared_feature)

            pooled_features.append(
                shared_feature.view(shared_feature.shape[0], -1))

        return torch.cat(pooled_features, dim=0) if pooled_features.__len__() > 0 else torch.tensor([]).to(pred_boxes.device)

    def get_instance_features(self, data_loader, class_id, max_instances, threshold, anchors=None):
        original_anchors = copy.deepcopy(self._get_anchors(class_id))
        if anchors is not None:
            self._set_anchors(class_id, anchors)

        model = build_network(model_cfg=cfg.MODEL, num_class=len(
            cfg.CLASS_NAMES), dataset=data_loader.dataset)
        model.cuda()
        model.load_params_from_file(
            filename=self.pretrained_model, to_cpu=self.dist)
        model.eval()
        if self.dist:
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
            max_instances //= commu_utils.get_world_size()

        instance_features = []
        total_instances = 0
        for _, data_dict in enumerate(data_loader):
            load_data_to_gpu(data_dict)

            pool_func = getattr(
                self, f'get_instance_features_{cfg.MODEL.NAME}')
            batch_features = pool_func(
                model, data_dict, class_id, threshold, anchors)

            instance_features.append(batch_features)
            n_batch_features = batch_features.shape[0]
            total_instances += n_batch_features
            if commu_utils.get_rank() == 0:
                print(
                    f'Extracting features: {total_instances}/{max_instances}', end='\r')
            if total_instances > max_instances:
                break

        self._set_anchors(class_id, original_anchors)
        instance_features = torch.cat(instance_features, dim=0)
        if commu_utils.get_world_size() > 1:
            instance_features = commu_utils.all_gather(
                instance_features)
            instance_features = torch.cat(instance_features, dim=0)
        return instance_features

    def fit_gmm(self, instance_features, n_components):
        if commu_utils.get_rank() == 0:
            self.logger.info(
                f'Estimating GMM parameters using {instance_features.shape[0]} samples...')
            gmm = [GMM(n_components).fit(instance_features)]
            self.logger.info('Finished GMM estimation.')
        else:
            gmm = [None]
        if commu_utils.get_world_size() > 1:
            dist.broadcast_object_list(gmm)
        return gmm[0]

    def linear_search(self, gmm, data_loader, class_id, search_range=0.2, step=0.05):
        current_anchors = copy.deepcopy(self._get_anchors(class_id))
        anchors_search_range = current_anchors * search_range
        anchor_ranges = [np.around(e, 1) for e in zip(
            current_anchors - anchors_search_range, current_anchors + anchors_search_range)]
        optimal_anchors = copy.deepcopy(current_anchors)

        for anchor_id in range(current_anchors.size):
            explored_anchors, explored_scores = [], []
            for cur_anchor_size in np.arange(*anchor_ranges[anchor_id], step):
                tmp_anchors = copy.deepcopy(current_anchors)
                tmp_anchors[anchor_id] = np.around(cur_anchor_size, 2)

                cur_instance_features = self.get_instance_features(
                    data_loader,
                    class_id,
                    cfg.SAILOR.TARGET.MAX_INSTANCES,
                    cfg.SAILOR.TARGET.THRESHOLD[class_id],
                    tmp_anchors
                )

                explored_anchors.append(tmp_anchors)
                explored_scores.append(gmm.pred(cur_instance_features))

                self.logger.info(
                    f'Anchor: {np.round(explored_anchors[-1], 2)}\tScore: {np.round(explored_scores[-1], 2)}')

            optimal_anchors[anchor_id] = explored_anchors[np.argmin(
                explored_scores)][anchor_id]
        self.logger.info(
            f'Linear search results for class {cfg.DATA_CONFIG_TARGET.CLASS_NAMES[class_id]}: {np.round(optimal_anchors, 2)}')
        self._set_anchors(class_id, optimal_anchors)

    def joint_optimization(self, gmm, data_loader, class_id, search_range=0.1):
        current_anchors = copy.deepcopy(self._get_anchors(class_id))
        bounds = np.array([(a-search_range, a+search_range)
                          for a in current_anchors])

        def func_to_optim(anc):
            cur_instance_features = self.get_instance_features(
                data_loader,
                class_id,
                cfg.SAILOR.TARGET.MAX_INSTANCES,
                cfg.SAILOR.TARGET.THRESHOLD[class_id],
                anc
            )
            score = gmm.pred(cur_instance_features)
            self.logger.info(
                f'Anchor: {np.round(anc, 2)}\tScore: {np.round(score, 2)}')
            return score

        self.logger.info(
            f'Staring ES with initial anchors {current_anchors} and bounds {bounds}')
        res = differential_evolution(func_to_optim, bounds, tol=1.)
        self.logger.info(f'Final result:\n{res}')
        self._set_anchors(class_id, res.x)


def calibrate(source_loader, target_loader, pretrained_model, dist, logger):
    calibrator = AnchorCalibrator(pretrained_model, dist, logger)
    for class_id, _ in enumerate(cfg.MODEL.DENSE_HEAD.ANCHOR_GENERATOR_CONFIG):
        # create reference feature database
        source_instance_features = calibrator.get_instance_features(
            source_loader,
            class_id,
            cfg.SAILOR.SOURCE.MAX_INSTANCES,
            cfg.SAILOR.SOURCE.THRESHOLD[class_id]
        )

        # fit a GMM to the data
        n_components = cfg.SAILOR.SOURCE.GMM_COMPONENTS[class_id]
        gmm = calibrator.fit_gmm(source_instance_features, n_components)

        if cfg.SAILOR.LINEAR_SEARCH:
            # perform linear search for each parameter separately
            calibrator.linear_search(gmm, target_loader, class_id)

        if cfg.SAILOR.JOINT_OPTIMIZATION:
            # perform joint optimization
            calibrator.joint_optimization(gmm, target_loader, class_id)
