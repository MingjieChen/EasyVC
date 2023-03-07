#!/bin/bash



splits=$1
dataset=$2
for split in $splits ; do
    
    echo "[whisper_ppg_medium feature extraction]: $split for $dataset"
    python3 ling_encoder/whisper_ppg/whisper_ppg_feature_extract.py \
        --ckpt ling_encoder/whisper_ppg/ckpt/medium.pt \
        --metadata data/$dataset/metadata.csv \
        --dump_dir dump/$dataset \
        --split $split \
        --max_workers 20
        --ext medium
done        

    
