# SAILOR
This repository contains the code for the paper:

**SAILOR: Scaling Anchors via Insights into Latent Object Representation**

[Dušan Malić](https://scholar.google.at/citations?user=EXovq6wAAAAJ), [Christian Fruhwirth-Reisinger](https://scholar.google.at/citations?user=Mg5Vlp8AAAAJ), [Horst Possegger](https://scholar.google.at/citations?user=iWPrl3wAAAAJ), [Horst Bischof](https://scholar.google.at/citations?user=_pq05Q4AAAAJ)

IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2023

[[arXiv]](https://arxiv.org/pdf/2210.07811.pdf)
| Source Only | Statistical Normalization | Random Object Scaling | SAILOR |
|:---:|:---:|:---:|:---:|
| <img src="https://files.icg.tugraz.at/f/082a95a4031d409ea546/?dl=1" width="200"/> | <img src="https://files.icg.tugraz.at/f/c2494ccfc9ca4fb8914a/?dl=1"  width="200"/> | <img src="https://files.icg.tugraz.at/f/3ab1285d5ced41afad06/?dl=1" width="200"/> | <img src="https://files.icg.tugraz.at/f/2ec8f8f0f3a74559b5ca/?dl=1" width="200"/> |

<details>
  <summary>Abstract</summary>
  
> LiDAR 3D object detection models are inevitably biased towards their training dataset. The detector clearly exhibits this bias when employed on a target dataset, particularly towards object sizes. However, object sizes vary heavily between domains due to, for instance, different labeling policies or geographical locations. State-of-the-art unsupervised domain adaptation approaches outsource methods to overcome the object size bias. Mainstream size adaptation approaches exploit target domain statistics, contradicting the original unsupervised assumption. Our novel unsupervised anchor calibration method addresses this limitation. Given a model trained on the source data, we estimate the optimal target anchors in a completely unsupervised manner. The main idea stems from an intuitive observation: by varying the anchor sizes for the target domain, we inevitably introduce noise or even remove valuable object cues. The latent object representation, perturbed by the anchor size, is closest to the learned source features only under the optimal target anchors. We leverage this observation for anchor size optimization. Our experimental results show that, without any retraining, we achieve competitive results even compared to state-of-the-art weakly-supervised size adaptation approaches. In addition, our anchor calibration can be combined with such existing methods, making them completely unsupervised.

</details>

## News
[2022-12-30] `SAILOR` is released.

## Installation
This repository is an extension of [`OpenPCDet v0.5.2`](https://github.com/open-mmlab/OpenPCDet/releases/tag/v0.5.2). Please follow the [installation](docs/INSTALL.md) and [getting started](docs/GETTING_STARTED.md) guide to set up an environment and extract the data.

## SAILOR
Anchor calibration process with `SAILOR` starts with a model pretrained on the source data. Afterwards, we use the source model in our calibration procedure.
*Note:* In the following, the configuration name is formatted as `<model name>_source-target.yaml`. By default, `source` data is used in the training script and `target` data in the evaluation script. Both `source` and `target` data are used during calibration.

### Waymo ⇾ KITTI 
Similarly to [Training & Testing](docs/GETTING_STARTED.md#training--testing), we train a source Waymo model as
```bash
sh scripts/dist_train.sh ${NUM_GPUS} \
    --cfg_file cfgs/sailor/PartA2_waymo-kitti.yaml
# or 
sh scripts/slurm_train.sh ${PARTITION} ${JOB_NAME} ${NUM_GPUS} \
    --cfg_file cfgs/sailor/PartA2_waymo-kitti.yaml
```

To evaluate a Waymo checkpoint on the KITTI *val* split:
```bash
python test.py \
    --cfg_file cfgs/sailor/PartA2_waymo-kitti.yaml \
    --batch_size ${BATCH_SIZE} \
    --ckpt ${CKPT}
```
or with multiple GPUs:
```bash
sh scripts/dist_test.sh ${NUM_GPUS} \
    --cfg_file ${CONFIG_FILE} \
    --batch_size ${BATCH_SIZE}
# or
sh scripts/slurm_test_mgpu.sh ${PARTITION} ${NUM_GPUS} \
    --cfg_file ${CONFIG_FILE} \
    --batch_size ${BATCH_SIZE}
```
With the pretrained model we can start the anchor calibration process:
```bash
python calibrate.py \
    --cfg_file cfgs/sailor/PartA2_waymo-kitti.yaml \
    --extra_tag anchor_calibration \
    --pretrained_model ${PRETRAINED_MODEL}
```
or using multiple GPUs:
```bash
python -m torch.distributed.launch --nproc_per_node=<num gpus> calibrate.py \
    --launcher pytorch \
    --cfg_file cfgs/sailor/PartA2_waymo-kitti.yaml \
    --extra_tag anchor_calibration \
    --pretrained_model ${PRETRAINED_MODEL}
```
In the table below, we report Averate Prevision (AP) of moderate difficulty on the KITTI *val* split. 
| Waymo ⇾ KITTI  | Car | Pedestrian | Cyclist | `cfg_file` | `pretrained_model` |
|---|:---:|:---:|:---:|:---:|:---:|
| Source Anchors | 23.94 | 59.99 | 52.32 | [PartA2_waymo-kitti.yaml](tools/cfgs/sailor/PartA2_waymo-kitti.yaml) | - |
| SAILOR | 58.02 | 61.60 | 53.04 | [PartA2_waymo-kitti.yaml](tools/cfgs/sailor/PartA2_waymo-kitti.yaml) | - |

Due to [Waymo Dataset License Agreement](https://waymo.com/open/terms/), we could not provide a pretrained model.

### nuScenes ⇾ KITTI
Pretraining/evaluation/calibration is performed equivalently as above using the provided [nuScenes ⇾ KITTI configuration](tools/cfgs/sailor/PartA2_nuscenes-kitti.yaml).

In the table below, we report Averate Prevision (AP) of moderate difficulty on the KITTI *val* split. 
| nuScenes ⇾ KITTI  | Car | Pedestrian | Cyclist | `cfg_file` | `pretrained_model` |
|---|:---:|:---:|:---:|:---:|:---:|
| Source Anchors | 24.54 | 15.13 | 24.23 | [PartA2_nuscenes-kitti.yaml](tools/cfgs/sailor/PartA2_nuscenes-kitti.yaml) | [nuscenes](https://files.icg.tugraz.at/f/07dd6e1dc57e429d9334/) |
| SAILOR | 55.17 | 11.60 | 23.12 | [PartA2_nuscenes-kitti.yaml](tools/cfgs/sailor/PartA2_nuscenes-kitti.yaml) | [nuscenes](https://files.icg.tugraz.at/f/07dd6e1dc57e429d9334/) |

## Acknowledgement
Our appreciation goes to the [OpenPCDet](https://github.com/open-mmlab/OpenPCDet/) authors and maintainers. This repository is the extension of their work and effort. Additionally, we are also thankful for open-sourcing [ST3D](https://github.com/CVMI-Lab/ST3D/) from where we adopted  a lot of valuable insights.

## Citation
```BibTeX
@inproceedings{wacv23sailor,
  title={{SAILOR: Scaling Anchors via Insights into Latent Object Representation}},
  author={Du\v{s}an Mali\'c and Christian Fruhwirth-Reisinger and Horst Possegger and Horst Bischof},
  booktitle={IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2023}
} 
```
