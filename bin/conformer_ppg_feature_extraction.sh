#!/bin/bash



splits=$1
dataset=$2

for split in $splits ; do
    
    echo "[conformer_ppg feature extraction]: $split for $dataset"
    python3 ling_encoder/conformer_ppg/conformer_ppg_feature_extract.py \
        --conformer_ppg_config ling_encoder/conformer_ppg/conformer_ppg_model/en_conformer_ctc_att/config.yaml\
        --conformer_ppg_ckpt ling_encoder/conformer_ppg/conformer_ppg_model/en_conformer_ctc_att/24epoch.pth \
        --metadata data/$dataset/metadata.csv \
        --dump_dir dump/$dataset \
        --split $split \
        --max_workers 20
done        

    
