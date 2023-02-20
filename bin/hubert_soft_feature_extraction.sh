#!/bin/bash



splits=$1
dataset=$2

for split in $splits ; do
    
    echo "[hubert_soft feature extraction]: $split for $dataset"
    python3 ling_encoder/hubert_soft/extract_features.py \
        --metadata data/$dataset/metadata.csv \
        --dump_dir dump/$dataset \
        --split $split \
        --max_workers 20
done        

    
