#!/bin/bash


db=downloads/vcc2020
data_dir=data/vcc2020



splits=("train_nodev" "dev" "eval")
spks=("SEF1" "SEF2" "SEM1" "SEM2" "TEF1" "TEF2" "TEM1" "TEM2" "TGM1" "TGF1" "TFM1" "TFF1" "TMM1" "TMF1")

for split in ${splits[*]}; do
    [ -e $data_dir/$split/wav.scp ] && rm $data_dir/$split/wav.scp
    [ -e $data_dir/$split/text ] && rm $data_dir/$split/text
    for spk in ${spks[*]}; do
        python3 dataset/vcc2020/data_prep.py \
            --db $db \
            --split $split \
            --spk $spk \
            --data_dir $data_dir
    done
done            



