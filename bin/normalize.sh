#!/bin/bash

conda=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7
conda_env=torch_1.7
source $conda/bin/activate $conda_env

for split in train_nodev_clean dev_clean ;do
    python preprocess/normalize.py \
            --stats_path dump/libritts/train_nodev_clean/mel/train_nodev_clean.npy \
            --dump_dir dump/libritts/ \
            --split $split \
            --metadata data/libritts/$split/metadata.csv
done            


