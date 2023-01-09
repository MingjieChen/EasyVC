#!/bin/bash

conda=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7
conda_env=torch_1.7
splits="train_nodev_clean dev_clean eval_clean"

script_dir=scripts/libritts/d_vector_speaker_level
[ ! -e $script_dir ]  && mkdir -p  $script_dir 
[ ! -e logs ] && mkdir logs
for split in $splits ; do
    
    echo "[d_vector speaker-level extraction]: $split for libritts"
    speakers=$(cat data/libritts/$split/speakers.txt)
    for spk in $speakers ; do 
        b=$script_dir/d_vector_speaker_level_${spk}.sh
        l=logs/enc_dec_d_vector_speaker_level.${spk}.log
        cat <<EOF > $b
#!/bin/bash
source $conda/bin/activate $conda_env
python3 speaker_encoder/d_vector/extract_speaker_embed.py \
    --d_vector_ckpt speaker_encoder/d_vector/d_vector_model/ckpt/pretrained_bak_5805000.pt \
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
