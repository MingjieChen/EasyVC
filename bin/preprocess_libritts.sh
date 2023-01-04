#!/bin/bash


conda=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7
conda_env=torch_1.7
source $conda/bin/activate $conda_env

splits="train_nodev_clean dev_clean eval_clean"

for split in $splits ; do
    
    echo "[preprocess]: $split for libritts"
    python3 preprocess/preprocess_libritts.py \
        --data_root downloads/LibriTTS \
        --scp_dir data/libritts \
        --metadata_dir data/libritts/$split/ \
        --split $split \
        --max_workers 20
done        

    
