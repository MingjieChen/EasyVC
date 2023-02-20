#!/bin/bash
conda=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7
conda_env=torch_1.9
source $conda/bin/activate $conda_env

config=configs/train_vqw2v_fastspeech2.yaml

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=virbr0
python train.py -c $config \
                -m vqw2v_uttdvec_none_fs2 \
                -e run \
                -l exp \
