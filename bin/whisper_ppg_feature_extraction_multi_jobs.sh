#!/bin/bash

conda=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7
conda_env=torch_1.9

dataset=vctk
splits="train_nodev_all dev_all eval_all"

script_dir=scripts/$dataset/whisper_ppg
[ ! -e $script_dir ]  && mkdir -p  $script_dir 
[ ! -e logs ] && mkdir logs
for split in $splits ; do
    
    echo "[whisper_ppgfeature extraction]: $split for $dataset"
    speakers=$(cat data/$dataset/$split/speakers.txt)
    for spk in $speakers ; do 
        b=$script_dir/whisper_ppg_feature_extraction_${split}_${spk}.sh
        l=logs/whisper_ppg_feature_extraction_${split}_${spk}.log
        cat <<EOF > $b
#!/bin/bash
source $conda/bin/activate $conda_env
python3 ling_encoder/whisper_ppg/whisper_ppg_feature_extract.py \
    --ckpt ling_encoder/whisper_ppg/ckpt/large-v2.pt \
    --metadata data/$dataset/$split/metadata.csv \
    --dump_dir dump/$dataset \
    --split $split \
    --max_workers 20 \
    --speaker $spk
EOF
    chmod +x $b
    submitjob -m 30000  $l $b
    echo "submitjob for $spk see log $l"
    done
done        
