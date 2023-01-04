#!/bin/bash

conda=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7
conda_env=torch_1.7
splits="train_nodev_clean dev_clean eval_clean"

script_dir=scripts/libritts/vqwav2vec
[ ! -e $script_dir ]  && mkdir -p  $script_dir 
[ ! -e logs ] && mkdir logs
for split in $splits ; do
    
    echo "[vqwav2vec feature extraction]: $split for libritts"
    speakers=$(cat data/libritts/$split/speakers.txt)
    for spk in $speakers ; do 
        b=$script_dir/vqwav2vec_feature_extraction_${spk}.sh
        l=logs/enc_dec_vqwav2vec_feature_extraction.${spk}.log
        cat <<EOF > $b
#!/bin/bash
source $conda/bin/activate $conda_env
python3 content_encoder/vqwav2vec/vqwav2vec_feature_extract.py \
    --vqwav2vec_ckpt content_encoder/vqwav2vec/vq-wav2vec_kmeans.pt \
    --metadata data/libritts/$split/metadata.csv \
    --dump_dir dump/libritts \
    --split $split \
    --max_workers 20 \
    --speaker $spk
EOF
    chmod +x $b
    submitjob -m 10000 $l $b
    echo "submitjob for $spk"
    break
    done
done        
