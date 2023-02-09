#!/bin/bash
conda=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7
conda_env=torch_1.7
source $conda/bin/activate $conda_env

echo "sge_task_id $SGE_TASK_ID"
python inference.py \
        --exp_dir exp/vqw2v_uttdvec_none_fastspeech2/first_train \
        --eval_list data/libritts/eval_clean/eval_list_oneshot_vc_small.json \
        --epochs 95 \
        --task oneshot_vc \
        --vocoder ppg_vc_hifigan  \
        --device cpu \
        --sge_task_id $SGE_TASK_ID \
        --sge_n_tasks 50
