#!/bin/bash



splits=$1
dataset=$2
for split in $splits ; do
    
    echo "[vqwav2vec feature extraction]: $split for libritts"
    python3 ling_encoder/contentvec_500/contentvec_500_feature_extract.py \
        --vqwav2vec_ckpt ling_encoder/contentvec_500/contentvec_500_model.pt \
        --metadata data/$dataset/metadata.csv \
        --dump_dir dump/$dataset \
        --split $split \
        --max_workers 20
done        

    
