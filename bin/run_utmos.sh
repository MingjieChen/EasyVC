#!/bin/bash

wav_dir=$1
out_csv=$2
conda=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7
conda_env=voicemos
source $conda/bin/activate $conda_env

python predict.py --mode predict_dir --inp_dir $wav_dir --bs 1 --out_path $out_csv
