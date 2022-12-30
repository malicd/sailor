import argparse
import datetime
import os
from pathlib import Path

import torch
import torch.nn as nn

from pcdet.config import cfg, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from pcdet.models import build_network
from eval_utils import eval_utils
from calib_utils.calib_utils import calibrate


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None,
                        help='specify the source config')

    parser.add_argument('--batch_size', type=int, default=None,
                        required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4,
                        help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str,
                        default='default', help='extra tag for this experiment')
    parser.add_argument('--pretrained_model', type=str,
                        default=None, help='pretrained_model')
    parser.add_argument(
        '--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888,
                        help='tcp port for distrbuted training')
    parser.add_argument('--fix_random_seed',
                        action='store_true', default=False, help='')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank for distributed training')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    # remove 'cfgs' and 'xxxx.yaml'
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])

    return args, cfg


def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    if args.fix_random_seed:
        common_utils.set_random_seed(666)

    output_dir = cfg.ROOT_DIR / 'output' / \
        cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' %
                             datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys(
    ) else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    # -----------------------create dataloader & network & optimizer---------------------------
    _, source_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist, workers=args.workers // 2,
        logger=logger,
        training=False,
    )

    _, target_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG_TARGET,
        class_names=cfg.DATA_CONFIG_TARGET.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist, workers=args.workers // 2,
        logger=logger,
        training=False,
    )

    # -----------------------start calibration---------------------------
    logger.info('**********************Start calibration %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    calibrate(
        source_loader=source_loader,
        target_loader=target_loader,
        pretrained_model=args.pretrained_model,
        dist=dist,
        logger=logger
    )
    logger.info('**********************End calibration %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    logger.info('**********************Start evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG_TARGET,
        class_names=cfg.DATA_CONFIG_TARGET.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist, workers=args.workers, logger=logger, training=False
    )
    eval_output_dir = output_dir / 'eval' / 'eval_with_calibrate'
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    model = build_network(model_cfg=cfg.MODEL, num_class=len(
        cfg.CLASS_NAMES), dataset=test_set)
    model.cuda()
    model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist)
    model.eval()
    if dist:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])

    # start evaluation
    eval_utils.eval_one_epoch(
        cfg, model, test_loader, 0, logger, dist_test=dist, result_dir=eval_output_dir
    )
    logger.info('**********************End evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


if __name__ == '__main__':
    main()
