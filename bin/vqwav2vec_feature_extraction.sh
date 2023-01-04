#!/bin/bash


conda=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7
conda_env=torch_1.7
source $conda/bin/activate $conda_env

splits="train_nodev_clean dev_clean eval_clean"

for split in $splits ; do
    
    echo "[vqwav2vec feature extraction]: $split for libritts"
    python3 content_encoder/vqwav2vec/vqwav2vec_feature_extract.py \
        --vqwav2vec_ckpt content_encoder/vqwav2vec/vq-wav2vec_kmeans.pt \
        --metadata data/libritts/metadata.csv \
        --dump_dir dump/libritts \
        --split $split \
        --max_workers 20
done        

    
