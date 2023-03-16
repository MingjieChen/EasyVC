#!/bin/bash


dataset=$1
splits=$2
feature_type=$3
stats_path=$4

for split in $splits ;do
    echo "running normalize for $feature_type  $dataset $split"
    python preprocess/normalize.py \
            --stats_path $stats_path \
            --dump_dir dump/$dataset \
            --split $split \
            --metadata data/$dataset/$split/metadata.csv \
            --feature_type  $feature_type
done            


