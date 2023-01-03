#!/bin/bash


conda=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7
conda_env=torch_1.7
source $conda/bin/activate $conda_env
root=$PWD
cd model/transformer_adversarial
speaker=1001
python extract_speaker_embed.py \
        /share/mini1/res/t/vc/studio/tiresyn-en/libritts/ParallelWaveGAN/egs/libritts/voc1/data/ \
       $root/dump/ppg-vc-spks \
       speaker_encoder/ckpt/pretrained_bak_5805000.pt \
       $speaker \
       /share/mini1/res/t/vc/studio/tiresyn-en/libritts/ParallelWaveGAN/egs/libritts/voc1


