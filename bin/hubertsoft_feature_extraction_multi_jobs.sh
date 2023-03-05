#!/bin/bash

conda=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7
conda_env=torch_1.9
splits="train_nodev_all dev_all eval_all"
dataset=vctk


script_dir=scripts/$dataset/hubert_soft
[ ! -e $script_dir ]  && mkdir -p  $script_dir 
[ ! -e logs ] && mkdir logs


for split in $splits ; do
    
    echo "[hubert_soft feature extraction]: $split for ${dataset}"
    speakers=$(cat data/$dataset/$split/speakers.txt)
    for spk in $speakers ; do 
        b=$script_dir/hubertsoft_feature_extraction_${split}_${spk}.sh
        l=logs/hubertsoft_feature_extraction_${split}_${spk}.log
        cat <<EOF > $b
#!/bin/bash
source $conda/bin/activate $conda_env
python3 ling_encoder/hubert_soft/extract_features.py \
    --metadata data/$dataset/$split/metadata.csv \
    --dump_dir dump/$dataset \
    --split $split \
    --max_workers 20 \
    --speaker $spk
EOF
    chmod +x $b
    submitjob -m 10000 $l $b
    echo "submitjob for [$dataset $split] [$spk], see log in $l"
    done
done        
