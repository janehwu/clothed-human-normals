#!/bin/bash

python train_normals.py \
    --cp rundir/tetratsdf/saved_models/checkpoint-500.pth.tar \
    --datasetroot /data/jwu/demo_dataset \
    --train_imgdirs datasetpaths_train/imgdirs_train_rp_aaron_posed_001.txt \
    --val_imgdirs datasetpaths_train/imgdirs_train_rp_aaron_posed_001.txt \
    --resolution 512 \
    --batch_size 1 \
    --save_interval 100 \
    --save_obj 100 \
    --end_epoch 100 \
    --lr 1e-4 \
    --use_pkl \
    --trial test

