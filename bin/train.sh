#!/bin/bash
conda=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7
conda_env=torch_1.7
source $conda/bin/activate $conda_env

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=virbr0
NCCL_DEBUG=INFO python train.py -c configs/train_vqw2v_fastspeech2.yaml
