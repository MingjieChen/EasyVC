#!/bin/bash

conda=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7
conda_env=torch_1.7
source $conda/bin/activate $conda_env

dataset=vctk
splits="train_nodev_all dev_all"
feature_type=fastspeech2_pitch_energy
stats_path=dump/$dataset/train_nodev_all/$feature_type/train_nodev_all.npy

for split in train_nodev_all dev_all ;do
    python preprocess/normalize.py \
            --stats_path $stats_path \
            --dump_dir dump/$dataset \
            --split $split \
            --metadata data/$dataset/$split/metadata.csv \
            --feature_type  $feature_type
done            


