#!/bin/bash

dataset=$1
train_split=$2
feature_type=$3

python preprocess/compute_statistics.py \
    --dump_dir dump/$dataset/ \
    --split $train_split \
    --metadata data/$dataset/$train_split/metadata.csv \
    --feature_type $feature_type
