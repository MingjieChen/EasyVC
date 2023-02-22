#!/bin/bash

# Copyright 2020 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)
# speaker setting
spks="all" # all or you can choose speakers e.g., "p225 p226 p227 ..."
sampling_rate=24000
# directory path setting
download_dir=downloads/vctk # directory to save database


train_set="train_nodev_$(echo "${spks}" | tr " " "_")" # name of training data directory
dev_set="dev_$(echo "${spks}" | tr " " "_")"           # name of development data directory
eval_set="eval_$(echo "${spks}" | tr " " "_")"         # name of evaluation data directory


echo "Stage -1: Data download"
dataset/vctk/data_download.sh "${download_dir}"

echo "Stage 0: Data preparation"
train_data_dirs=""
dev_data_dirs=""
eval_data_dirs=""
# if set to "all", use all of the speakers in the corpus
if [ "${spks}" = "all" ]; then
    # NOTE(kan-bayashi): p315 will not be used since it lacks txt data
    spks=$(find "${download_dir}/VCTK-Corpus/wav48" \
        -maxdepth 1 -name "p*" -exec basename {} \; | sort | grep -v p315)
fi
for spk in ${spks}; do
    dataset/vctk/data_prep.sh \
        --fs $sampling_rate \
        --train_set "train_nodev_${spk}" \
        --dev_set "dev_${spk}" \
        --eval_set "eval_${spk}" \
        "${download_dir}/" "${spk}" data/vctk
    train_data_dirs+=" data/vctk/train_nodev_${spk}"
    dev_data_dirs+=" data/vctk/dev_${spk}"
    eval_data_dirs+=" data/vctk/eval_${spk}"
done
# shellcheck disable=SC2086
dataset/libritts/combine_data.sh "data/vctk/${train_set}" ${train_data_dirs}
# shellcheck disable=SC2086
dataset/libritts/combine_data.sh "data/vctk/${dev_set}" ${dev_data_dirs}
# shellcheck disable=SC2086
dataset/libritts/combine_data.sh "data/vctk/${eval_set}" ${eval_data_dirs}
