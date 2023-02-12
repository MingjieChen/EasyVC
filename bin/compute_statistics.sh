#!/bin/bash

conda=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7
conda_env=torch_1.7
source $conda/bin/activate $conda_env

python preprocess/compute_statistics.py \
    --dump_dir dump/libritts/ \
    --split train_nodev_clean \
    --metadata data/libritts/train_nodev_clean/metadata.csv
