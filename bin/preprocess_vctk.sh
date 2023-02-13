#!/bin/bash


conda=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7
conda_env=torch_1.7
source $conda/bin/activate $conda_env

splits="train_nodev_all dev_all eval_all"

for split in $splits ; do
    
    echo "[preprocess]: $split for vctk"
    python3 preprocess/preprocess_vctk.py \
        --data_root downloads/vctk \
        --scp_dir data/vctk \
        --metadata_dir data/vctk/$split/ \
        --split $split \
        --max_workers 20
done        

    
