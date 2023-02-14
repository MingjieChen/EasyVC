#!/bin/bash

conda=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7
conda_env=torch_1.7
source $conda/bin/activate $conda_env

for split in train_nodev_all dev_all ;do
    python preprocess/normalize.py \
            --stats_path dump/vctk/train_nodev_all/mel/train_nodev_all.npy \
            --dump_dir dump/vctk/ \
            --split $split \
            --metadata data/vctk/$split/metadata.csv
done            


