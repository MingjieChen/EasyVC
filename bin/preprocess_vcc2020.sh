#!/bin/bash

conda=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7
conda_env=torch_1.9
source $conda/bin/activate $conda_env


splits="train_nodev dev eval"

for split in $splits ; do
    
    echo "[preprocess]: $split for vcc2020"
    python3 preprocess/preprocess_vcc2020.py \
        --data_root downloads/vcc2020 \
        --scp_dir data/vcc2020 \
        --metadata_dir data/vcc2020/$split/ \
        --split $split \
        --max_workers 20
done        

    
