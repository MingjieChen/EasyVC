#!/bin/bash



splits=$1
dataset=$2
for split in $splits ; do
    
    echo "[vqwav2vec feature extraction]: $split for libritts"
    python3 ling_encoder/vqwav2vec/vqwav2vec_feature_extract.py \
        --vqwav2vec_ckpt ling_encoder/vqwav2vec/vq-wav2vec_kmeans.pt \
        --metadata data/$dataset/metadata.csv \
        --dump_dir dump/$dataset \
        --split $split \
        --max_workers 20
done        

    
