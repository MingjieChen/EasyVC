#!/bin/bash

task=vc
dataset=vcc2020
split=eval
eval_list=eval_list_crosslingual_a2a_vc_oneshot.json
n_trg_spk_samples=1
n_src_spk_samples=25
n_eval_spks=100000


. ./bin/parse_options.sh || exit 1;


if [ "$dataset" != "vcc2020" ]; then
    if [ ! -e data/$dataset/$split/metadata_with_wrd.csv  ]; then
        echo "update eval metadata with wrd"
        # update eval metadata with text transcriptions
        python3 evaluation/update_metadata_${dataset}.py \
                    --metadata_path  data/${dataset}/${split}/metadata.csv \
                    --out_path data/${dataset}/${split}/metadata_with_wrd.csv
    fi
else
    if [ ! -e data/$dataset/$split/metadata_with_wrd.csv  ]; then
        cd data/$dataset/$split; ln -s  metadata.csv metadata_with_wrd.csv; cd ../../../
    fi    
fi                

        
echo "done!"

echo "generate eval list for a2a vc"
# generate eval list
opts=""
if [ "$dataset" == "vcc2020" ]; then
    opts+=" --src_speakers SEF1 SEF2 SEM1 SEM2\
            --trg_speakers TFF1 TFM1 TGM1 TGF1 TMM1 TMF1
        "
fi        
python3 evaluation/generate_eval_list.py \
        --task $task \
        --split $split \
        --spk_enc utt_dvec \
        --speakers_path data/$dataset/$split/speakers.txt \
        --eval_metadata_path data/$dataset/$split/metadata_with_wrd.csv \
        --eval_list_out_path data/$dataset/$split/${eval_list} \
        --n_samples_per_trg_speaker $n_trg_spk_samples \
        --n_eval_speakers $n_eval_spks \
        --n_samples_per_src_speaker $n_src_spk_samples \
        $opts
echo "done!"
