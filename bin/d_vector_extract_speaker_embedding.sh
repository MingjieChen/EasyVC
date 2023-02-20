#!/bin/bash


splits=$1
dataset=$2

for split in $splits ; do
    
    echo "[d_vector speaker-level extraction]: $split for $dataset"
    python3 speaker_encoder/d_vector/extract_speaker_embed.py \
        --d_vector_ckpt speaker_encoder/d_vector/d_vector_model/ckpt/pretrained_bak_5805000.pt \
        --metadata data/$dataset/$split/metadata.csv \
        --dump_dir dump/$dataset \
        --split $split \
        --max_workers 20
done        


