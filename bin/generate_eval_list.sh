#!/bin/bash

task=vc
dataset=vctk
split=eval_all
eval_list=eval_list_m2m_vc_small_oneshot.json
n_trg_spk_samples=1
n_src_spk_samples=4
n_eval_spks=10

. ./parse_options.sh || exit 1;

if [ ! -e data/$dataset/$split/metadata_with_wrd.csv  ]; then
    echo "update eval metadata with wrd"
    # update eval metadata with text transcriptions
    python3 evaluation/update_metadata_${dataset}.py \
            --metadata_path  data/${dataset}/${split}/metadata.csv \
            --out_path data/${dataset}/${split}/metadata_with_wrd.csv
fi            

        
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
        --n_samples_per_trg_speaker $n_trg_spk_samples \
        --n_eval_speakers $n_eval_spks \
        --n_samples_per_src_speaker $n_src_spk_samples
echo "done!"
