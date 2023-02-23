#!/bin/bash


splits=$1
dataset=$2
for split in $splits ; do
    
    echo "[ecapa-tdnn utterance-level extraction]: $split for $dataset"
    python3 speaker_encoder/ecapa_tdnn/extract_utter_embed.py \
        --metadata data/$dataset/$split/metadata.csv \
        --dump_dir dump/$dataset \
        --split $split \
        --max_workers 20
done        


