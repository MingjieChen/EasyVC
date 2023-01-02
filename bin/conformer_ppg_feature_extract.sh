#!/bin/bash

conda=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7
conda_env=torch_1.7
source $conda/bin/activate $conda_env

#export PATH=/share/mini1/sw/std/cuda/cuda11.1/bin:$PATH
#export CUDA_HOME=/share/mini1/sw/std/cuda/cuda11.1/
#export LD_LIBRARY_PATH=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7/envs/StyleSpeech/lib:/share/mini1/sw/std/cuda/cuda11.1/lib64:$LD_LIBRARY_PATH
root=$PWD
cd model/transformer_adversarial
python extract_conformer_ppgbnf.py \
    --data_root $root/vctk/VCTK-Corpus/wav48 \
    --scp_dir /share/mini1/res/t/vc/studio/tiresyn-en/vctk/ParallelWaveGAN/egs/vctk/voc1/data \
    --speakers_path $root/speakers.json \
    --config_path preprocess.yaml \
    --mel_dir $root/dump/transformer_adversarial/mel \
    --out_dir $root/dump/transformer_adversarial/trim_ppg_bnf
