#!/bin/bash


conda=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7
conda_env=torch_1.7
source $conda/bin/activate $conda_env

splits="train_nodev_clean dev_clean eval_clean"

for split in $splits ; do
    
    echo "[conformer_ppg feature extraction]: $split for libritts"
    python3 content_encoder/conformer_ppg/conformer_ppg_feature_extract.py \
        --conformer_ppg_config content_encoder/conformer_ppg/conformer_ppg_model/en_conformer_ctc_att/config.yaml\
        --conformer_ppg_ckpt content_encoder/conformer_ppg/conformer_ppg_model/en_conformer_ctc_att/24epoch.pth \
        --metadata data/libritts/metadata.csv \
        --dump_dir dump/libritts \
        --split $split \
        --max_workers 20
done        

    
