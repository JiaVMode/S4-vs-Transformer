#!/bin/bash

conda activate S4

# Settings
export CUDA_VISIBLE_DEVICES=0,1



# 示例运行方式: ./scripts/run.sh

python train.py --config configs/config_s4.yaml

python train.py --config configs/config_transformer.yaml









