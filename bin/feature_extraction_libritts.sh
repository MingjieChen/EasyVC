#!/bin/bash
conda=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7
conda_env=torch_1.7
source $conda/bin/activate $conda_env

splits="train_nodev_clean dev_clean eval_clean"


for split in $splits ; do
    
    echo "[feature extraction]: $split for libritts"
    python3 preprocess/feature_extraction.py \
        --metadata data/libritts/$split/metadata.csv \
        --dump_dir dump/libritts \
        --config_path  configs/preprocess_libritts.yaml \
        --split $split \
        --max_workers 20
done        
