#!/bin/bash

conda=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7
conda_env=torch_1.7
source $conda/bin/activate $conda_env
PYTHON=$conda/envs/$conda_env/bin/python

audio_dir=/share/mini1/res/t/vc/studio/timap-en/vctk/vctk/VCTK-Corpus/wav48
root=/share/mini1/res/t/vc/studio/timap-en/vctk
#python model/new_stargan_vc/wav_dataloader.py $audio_dir/SEF1/E30001.wav
cd /share/mini1/res/t/vc/studio/timap-en/vctk/fairseq/examples/wav2vec
python  vqwv2vec_feat_extract.py $audio_dir $root/dump/pwg_trim_vqw2v_feat/ /share/mini1/res/t/vc/studio/tiresyn-en/vctk/ParallelWaveGAN/egs/vctk/voc1/data
