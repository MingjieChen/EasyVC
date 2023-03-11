#!/bin/bash

conda=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7
conda_env=torch_1.7

# setup

dataset=vctk
config=configs/preprocess_bigvgan_mel.yaml
feature_type=bigvgan_mel
splits="train_nodev_all dev_all"

script_dir=scripts/$dataset/preprocess

[ ! -e $script_dir ]  && mkdir -p  $script_dir 

for split in $splits ; do
    
    echo "[feature extraction]: $split $dataset $feature_type"
    speakers=$(cat data/$dataset/$split/speakers.txt)
    for spk in $speakers ; do 
        b=$script_dir/feature_extraction_${feature_type}_${split}_${spk}.sh
        l=logs/feature_extraction_${feature_type}_${split}_${spk}.log
        cat <<EOF > $b
#!/bin/bash
source $conda/bin/activate $conda_env
python3 feature_extraction.py \
    --metadata data/$dataset/$split/metadata.csv \
    --dump_dir dump/$dataset \
    --config_path  $config \
    --split $split \
    --max_workers 20 \
    --feature_type $feature_type \
    --speaker $spk
EOF
    chmod +x $b
    submitjob -m 10000 $l $b
    echo "submitjob for $dataset $split  $spk $feature_type see log $l"
    done
done        
