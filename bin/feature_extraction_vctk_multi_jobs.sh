#!/bin/bash

conda=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7
conda_env=torch_1.7

# setup

dataset=vctk
config=configs/preprocess_vctk_ppgvc_mel.yaml
mel_type=ppgvc_mel
splits="train_nodev_all dev_all"

script_dir=scripts/$dataset/preprocess

[ ! -e $script_dir ]  && mkdir -p  $script_dir 

for split in $splits ; do
    
    echo "[feature extraction]: $split for $dataset"
    speakers=$(cat data/$dataset/$split/speakers.txt)
    for spk in $speakers ; do 
        b=$script_dir/feature_extraction_${split}_${spk}.sh
        l=logs/feature_extraction_${split}.${spk}.log
        cat <<EOF > $b
#!/bin/bash
source $conda/bin/activate $conda_env
python3 preprocess/feature_extraction.py \
    --metadata data/$dataset/$split/metadata.csv \
    --dump_dir dump/$dataset \
    --config_path  $config \
    --split $split \
    --max_workers 20 \
    --mel_type $mel_type \
    --speaker $spk
EOF
    chmod +x $b
    submitjob -m 10000 $l $b
    echo "submitjob for $dataset $split  $spk"
    done
done        