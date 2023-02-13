#!/bin/bash
conda=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7
conda_env=torch_1.7
source $conda/bin/activate $conda_env

splits="train_nodev_all dev_all eval_all"


for split in $splits ; do
    
    echo "[feature extraction]: $split for vctk"
    python3 preprocess/feature_extraction.py \
        --metadata data/vctk/$split/metadata.csv \
        --dump_dir dump/vctk \
        --config_path  configs/preprocess_vctk.yaml \
        --split $split \
        --max_workers 20
done        
