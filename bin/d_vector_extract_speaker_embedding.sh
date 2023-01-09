#!/bin/bash


conda=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7
conda_env=torch_1.7
source $conda/bin/activate $conda_env
splits="train_nodev_clean dev_clean eval_clean"

for split in $splits ; do
    
    echo "[d_vector speaker-level extraction]: $split for libritts"
    python3 speaker_encoder/d_vector/extract_speaker_embed.py \
        --d_vector_ckpt speaker_encoder/d_vector/d_vector_model/ckpt/pretrained_bak_5805000.pt \
        --metadata data/libritts/$split/metadata.csv \
        --dump_dir dump/libritts \
        --split $split \
        --max_workers 20
done        


