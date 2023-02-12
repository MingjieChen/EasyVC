#!/bin/bash

#echo "update eval metadata with wrd"
# update eval metadata with text transcriptions
#python3 evaluation/update_metadata_libritts.py \
#        --metadata_path  data/libritts/eval_clean/metadata.csv \
#        --out_path data/libritts/eval_clean/metadata_with_wrd.csv
#echo "done!"

echo "generate eval list for a2a vc"
# generate eval list
python3 evaluation/generate_eval_list.py \
        --task vc \
        --split eval_clean \
        --spk_enc utt_dvec \
        --speakers_path data/libritts/eval_clean/speakers.txt \
        --eval_metadata_path data/libritts/eval_clean/metadata_with_wrd.csv \
        --eval_list_out_path data/libritts/eval_clean/eval_list_a2a_vc_small.json \
        --n_samples_per_trg_speaker 10 \
        --n_eval_speakers 10 \
        --n_samples_per_src_speaker 4
echo "done!"
