#!/bin/bash


# speaker setting
part="clean" # "clean" or "all"
             # if set to "clean", use only clean data
             # if set to "all", use clean + other data

# directory path setting
download_dir=downloads # directory to save database
dumpdir=dump           # directory to dump features


train_set="train_nodev_${part}" # name of training data directory
dev_set="dev_${part}"           # name of development data directory
eval_set="eval_${part}"         # name of evaluation data directory

set -euo pipefail
    
echo "Stage -1: Data download"
local/data_download.sh "${download_dir}"


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
    local/data_prep.sh "${download_dir}/LibriTTS" \
        "${train_part}" data "${download_dir}/LibriTTSLabel"
    train_data_dirs+=" data/${train_part}"
done
for dev_part in ${dev_parts}; do
    local/data_prep.sh "${download_dir}/LibriTTS" \
        "${dev_part}" data "${download_dir}/LibriTTSLabel"
    dev_data_dirs+=" data/${dev_part}"
done
for eval_part in ${eval_parts}; do
    local/data_prep.sh "${download_dir}/LibriTTS" \
        "${eval_part}" data "${download_dir}/LibriTTSLabel"
    eval_data_dirs+=" data/${eval_part}"
done
# shellcheck disable=SC2086
utils/combine_data.sh "data/${train_set}" ${train_data_dirs}
# shellcheck disable=SC2086
utils/combine_data.sh "data/${dev_set}" ${dev_data_dirs}
# shellcheck disable=SC2086
utils/combine_data.sh "data/${eval_set}" ${eval_data_dirs}
