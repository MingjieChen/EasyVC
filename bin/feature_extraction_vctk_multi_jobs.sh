#!/bin/bash

conda=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7
conda_env=torch_1.7
splits="train_nodev_all dev_all eval_all"

script_dir=scripts/vctk/preprocess
[ ! -e $script_dir ]  && mkdir -p  $script_dir 

for split in $splits ; do
    
    echo "[feature extraction]: $split for vctk"
    speakers=$(cat data/vctk/$split/speakers.txt)
    for spk in $speakers ; do 
        b=$script_dir/feature_extraction_${spk}.sh
        l=logs/enc_dec_feature_extraction.${spk}.log
        cat <<EOF > $b
#!/bin/bash
source $conda/bin/activate $conda_env
python3 preprocess/feature_extraction.py \
    --metadata data/vctk/$split/metadata.csv \
    --dump_dir dump/vctk \
    --config_path  configs/preprocess_vctk.yaml \
    --split $split \
    --max_workers 20 \
    --speaker $spk
EOF
    chmod +x $b
    submitjob -m 10000 -M8 $l $b
    echo "submitjob for $spk"
    done
done        
