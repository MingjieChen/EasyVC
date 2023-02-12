#!/bin/bash

conda=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7
conda_env=speechbrain
source $conda/bin/activate $conda_env

export PATH=/share/mini1/sw/std/cuda/cuda11.1/bin:$PATH
export CUDA_HOME=/share/mini1/sw/std/cuda/cuda11.1/
export LD_LIBRARY_PATH=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7/envs/StyleSpeech/lib:/share/mini1/sw/std/cuda/cuda11.1/lib64:$LD_LIBRARY_PATH
#exp_name=0127_mel_Gsplit_cyc2_id1_adv1_lr

test_csv=
wer_file=
output_folder=

python evaluation/speechbrain_asr.py  evaluation/speechbrain_asr.yaml  \
        --test_csv=[$test_csv]  \
        --wer_file=$wer_file \
        --output_folder=$output_folder
