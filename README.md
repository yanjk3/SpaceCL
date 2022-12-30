# SpaceCL

Official PyTorch implementation of ICPR2022 paper “Space-correlated Contrastive Representation Learning with Multiple Instances”

## Environments
- python 3.7
- pytorch 1.6.0
- cuda 10.2

## Pre-training
To pre-train SpaceCL on COCO for 200 epochs, you can follow this training script:

```
python main_spacecl.py -a resnet50 --lr 0.05 --batch-size 256 \
       --dist-url tcp://localhost:10005 --multiprocessing-distributed --world-size 1 --rank 0 \
       --mlp --moco-t 0.2 --aug-plus --cos --epochs 200 \
       --weight-type iou --gamma 1.0 \
       /path/to/COCO2017/trainingset/
```

## Transferring to Object Detection
First, you should convert the pre-trained model to a standard R50 model:
```
python transfer2R50.py /path/to/input/checkpoint /path/to/output/checkpoint
```

Then, you can use the official [mmdetection](https://github.com/open-mmlab/mmdetection) to train the detection model. 
Please refer to mmdetection for more details.
For example, to train a mask r-cnn on COCO for 1x schedule, you can follow this training script:
```
cd mmdetection
sh ./tools/dist_train.sh configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py \ 
    --options model.init_cfg.checkpoint=/path/to/output/checkpoint 8
```

## Acknowledgement 
- This repository is heavily based on [MoCo](https://github.com/facebookresearch/moco) and [ReSim](https://github.com/Tete-Xiao/ReSim).

- If you use this paper/code in your research, please consider citing us:
```
@inproceedings{song2022space,
  title={Space-correlated Contrastive Representation Learning with Multiple Instances},
  author={Song, Danming and Gao, Yipeng and Yan, Junkai and Sun, Wei and Zheng, Wei-Shi},
  booktitle={International Conference on Pattern Recognition},
  year={2022},
}
```
