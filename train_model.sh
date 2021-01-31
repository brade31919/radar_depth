#!/usr/bin/env bash
python main.py \
    --arch resnet18_multistage_uncertainty_fixs \
    --data nuscenes \
    --modality rgbd \
    --decoder upproj \
    -j 16 \
    --epochs 20 \
    -b 8 \
    --num-samples 50 \
    --max-depth 80 \
    --sparsifier radar
    # --resume ./results/sparse_to_dense/nuscenes.sparsifier\=radar.samples\=50.modality\=rgbd.arch\=resnet18_latefusion.decoder\=upproj.criterion\=l1.lr\=0.01.bs\=16.pretrained\=True/checkpoint-0.pth.tar \
    
              
              