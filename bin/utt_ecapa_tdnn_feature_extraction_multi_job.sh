#!/bin/bash

conda=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7
conda_env=speechbrain

# setup
dataset=vctk
splits="train_nodev_all dev_all"
script_dir=scripts/$dataset/ecapa_tdnn_utterance_level
[ ! -e $script_dir ]  && mkdir -p  $script_dir 
[ ! -e logs ] && mkdir logs
for split in $splits ; do
    
    echo "[ecapa_tdnn utterance-level extraction]: $split for $dataset"
    speakers=$(cat data/$dataset/$split/speakers.txt)
    for spk in $speakers ; do 
        b=$script_dir/ecapa_tdnn_utterance_level_${split}_${spk}.sh
        l=logs/ecapa_tdnn_utterance_level_${split}_${spk}.log
        cat <<EOF > $b
#!/bin/bash
source $conda/bin/activate $conda_env
python3 speaker_encoder/ecapa_tdnn/extract_utter_embed.py \
    --metadata data/$dataset/$split/metadata.csv \
    --dump_dir dump/$dataset \
    --split $split \
    --max_workers 20 \
    --speaker $spk
EOF
    chmod +x $b
    submitjob -m 10000 $l $b
    echo "submitjob for $spk see log $l"
    done
done        
