#!/bin/bash

dataset=$1
feature_type=$2
splits=$3
config=configs/preprocess_${feature_type}.yaml


for split in $splits ; do
    echo "[feature extraction]: $split $dataset $feature_type"
    python3 feature_extraction.py \
        --metadata data/$dataset/$split/metadata.csv \
        --dump_dir dump/$dataset \
        --config_path  $config\
        --split $split \
        --feature_type $feature_type \
        --max_workers 20
done        
