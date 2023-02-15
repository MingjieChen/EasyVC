#!/bin/bash
conda=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7
conda_env=torch_1.7
source $conda/bin/activate $conda_env

dataset=vctk
config=configs/preprocess_vctk_ppgvc_mel.yaml
feature_type=ppgvc_f0
splits="train_nodev_all dev_all"


for split in $splits ; do
    
    echo "[feature extraction]: $split $dataset $feature_type"
    python3 feature_extraction.py \
        --metadata data/$dataset/$split/metadata.csv \
        --dump_dir dump/$dataset \
        --config_path  $config\
        --split $split \
        --feature_type $feature_type \
        --max_workers 20
done        
