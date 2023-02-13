#!/bin/bash

# Copyright 2020 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)
#conda=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7

# speaker setting
part="clean" # "clean" or "all"
             # if set to "clean", use only clean data
             # if set to "all", use clean + other data

# directory path setting
download_dir=downloads # directory to save database
dumpdir=dump/libritts_clean           # directory to dump features
datadir=data/libritts   # directory to store meta data (wav.scp, segments)

train_set="train_nodev_${part}" # name of training data directory
dev_set="dev_${part}"           # name of development data directory
eval_set="eval_${part}"         # name of evaluation data directory

set -euo pipefail
    
echo "Stage -1: Data download"
dataset/libritts/data_download.sh "${download_dir}"


echo "Stage 0: Data preparation"
if [ "${part}" = "clean" ]; then
    train_parts="train-clean-100 train-clean-360"
    dev_parts="dev-clean"
    eval_parts="test-clean"
elif [ "${part}" = "all" ]; then
    train_parts="train-clean-100 train-clean-360 train-other-500"
    dev_parts="dev-clean dev-other"
    eval_parts="test-clean test-other"
else
    echo "You must select from all or clean." >&2; exit 1;
fi
train_data_dirs=""
dev_data_dirs=""
eval_data_dirs=""


for train_part in ${train_parts}; do
    dataset/libritts/data_prep.sh "${download_dir}/LibriTTS" \
        "${train_part}" $datadir "${download_dir}/LibriTTSLabel"
    train_data_dirs+=" ${datadir}/${train_part}"
done
for dev_part in ${dev_parts}; do
    dataset/libritts/data_prep.sh "${download_dir}/LibriTTS" \
        "${dev_part}" $datadir "${download_dir}/LibriTTSLabel"
    dev_data_dirs+=" ${datadir}/${dev_part}"
done
for eval_part in ${eval_parts}; do
    dataset/libritts/data_prep.sh "${download_dir}/LibriTTS" \
        "${eval_part}" $datadir "${download_dir}/LibriTTSLabel"
    eval_data_dirs+=" ${datadir}/${eval_part}"
done
# shellcheck disable=SC2086
dataset/libritts/combine_data.sh "${datadir}/${train_set}" ${train_data_dirs}
# shellcheck disable=SC2086
dataset/libritts/combine_data.sh "${datadir}/${dev_set}" ${dev_data_dirs}
# shellcheck disable=SC2086
dataset/libritts/combine_data.sh "${datadir}/${eval_set}" ${eval_data_dirs}
