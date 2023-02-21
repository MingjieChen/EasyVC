#!/bin/bash

task=vc
dataset=vctk
split=eval_all
eval_list=eval_list_m2m_vc_small.json
echo "update eval metadata with wrd"
# update eval metadata with text transcriptions
python3 evaluation/update_metadata_${dataset}.py \
        --metadata_path  data/${dataset}/${split}/metadata.csv \
        --out_path data/${dataset}/${split}/metadata_with_wrd.csv

        
echo "done!"

echo "generate eval list for a2a vc"
# generate eval list
python3 evaluation/generate_eval_list.py \
        --task $task \
        --split $split \
        --spk_enc utt_dvec \
        --speakers_path data/$dataset/$split/speakers.txt \
        --eval_metadata_path data/$dataset/$split/metadata_with_wrd.csv \
        --eval_list_out_path data/$dataset/$split/${eval_list} \
        --n_samples_per_trg_speaker 10 \
        --n_eval_speakers 10 \
        --n_samples_per_src_speaker 4
echo "done!"
