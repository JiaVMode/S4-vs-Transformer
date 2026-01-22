#!/bin/bash

# Settings
export CUDA_VISIBLE_DEVICES=0,1

python train.py --config configs/config_s4.yaml

python train.py --config configs/config_transformer.yaml









