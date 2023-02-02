#!/bin/bash

conda=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7
conda_env=torch_1.9
splits="train_nodev_clean dev_clean eval_clean"

script_dir=scripts/libritts/hubert_km100
[ ! -e $script_dir ]  && mkdir -p  $script_dir 
[ ! -e logs ] && mkdir logs
for split in $splits ; do
    
    echo "[hubert_km100 feature extraction]: $split for libritts"
    speakers=$(cat data/libritts/$split/speakers.txt)
    for spk in $speakers ; do 
        b=$script_dir/hubertkm100_feature_extraction_${spk}.sh
        l=logs/enc_dec_hubertkm100_feature_extraction.${spk}.log
        cat <<EOF > $b
#!/bin/bash
source $conda/bin/activate $conda_env
python3 ling_encoder/hubert_km100/extract_features.py \
    --metadata data/libritts/$split/metadata.csv \
    --dump_dir dump/libritts \
    --split $split \
    --max_workers 20 \
    --speaker $spk
EOF
    chmod +x $b
    submitjob -m 10000 $l $b
    echo "submitjob for $spk"
    done
done        
