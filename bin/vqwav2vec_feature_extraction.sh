#!/bin/bash



splits=$1
dataset=$2

if [ ! -e ling_encoder/vqwav2vec/vq-wav2vec_kmeans.pt ]; then 
    echo "downloading vqwav2vec model checkpoint"
    mkdir -p ling_encoder/vqwav2vec 
    cd ling_encoder/vqwav2vec
    wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/vq-wav2vec_kmeans.pt
    cd ../..
    echo "done!"
fi   
for split in $splits ; do
    
    echo "[vqwav2vec feature extraction]: $split for $dataset"
    python3 ling_encoder/vqwav2vec/vqwav2vec_feature_extract.py \
        --vqwav2vec_ckpt ling_encoder/vqwav2vec/vq-wav2vec_kmeans.pt \
        --metadata data/$dataset/$split/metadata.csv \
        --dump_dir dump/$dataset \
        --split $split \
        --max_workers 20
done        

    
