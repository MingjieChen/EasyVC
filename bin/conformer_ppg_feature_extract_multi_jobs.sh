#!/bin/bash

conda=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7
conda_env=torch_1.7
dataset=vctk
splits="train_nodev_all dev_all eval_all"

script_dir=scripts/$dataset/conformer_ppg
[ ! -e $script_dir ]  && mkdir -p  $script_dir 
[ ! -e logs ] && mkdir logs
for split in $splits ; do
    
    echo "[conformer_ppg feature extraction]: $split for $dataset"
    speakers=$(cat data/$dataset/$split/speakers.txt)
    for spk in $speakers ; do 
        b=$script_dir/conformer_ppg_feature_extraction_${split}_${spk}.sh
        l=logs/enc_dec_conformer_ppg_feature_extraction_${split}_${spk}.log
        cat <<EOF > $b
#!/bin/bash
source $conda/bin/activate $conda_env
python3 ling_encoder/conformer_ppg/conformer_ppg_feature_extract.py \
    --conformer_ppg_config ling_encoder/conformer_ppg/conformer_ppg_model/en_conformer_ctc_att/config.yaml\
    --conformer_ppg_ckpt ling_encoder/conformer_ppg/conformer_ppg_model/en_conformer_ctc_att/24epoch.pth \
    --metadata data/$dataset/$split/metadata.csv \
    --dump_dir dump/$dataset \
    --split $split \
    --max_workers 20 \
    --speaker $spk
EOF
    chmod +x $b
    submitjob -m 10000 $l $b
    echo "submitjob for $spk"
    done
done        
