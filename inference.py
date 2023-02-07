import random
import yaml
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import sys
import argparse
import os
import json
import glob
import soundfile as sf
import csv
from tqdm import tqdm
from scipy.io import wavfile
import resampy
from ling_encoder.interface import *
from speaker_encoder.interface import *
from decoder.interface import *
from vocoder.interface import *

def load_wav(path, sample_rate = 16000):
    sr, x = wavfile.read(path)
    signed_int16_max = 2**15
    if x.dtype == np.int16:
        x = x.astype(np.float32) / signed_int16_max
    if sr != sample_rate:
        x = resampy.resample(x, sr, sample_rate)
    x = np.clip(x, -1.0, 1.0)
    return x


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--exp_dir', type = str)
    parser.add_argument('--eval_list',type = str)
    parser.add_argument('--device', type = str, default = 'cpu')
    # encoder decoder configs
    parser.add_argument('--enc_dec_config', type = str)
    #exp 
    parser.add_argument('--epochs', type = int)
    parser.add_argument('--task', type = str) 
    # vocoder
    parser.add_argument('--vocoder', type = str, default = 'ppg_vc_hifigan')
    # sge task 
    parser.add_argument('--sge_n_jobs', type = int, default = 1)
    parser.add_argument('--seg_job_idx', type = int, default = 0)

    # arguments
    args = parser.parse_args()
    print(args)

    # load encoder decoder configs
    with open(args.enc_dec_config) as f:
        enc_dec_config = yaml.safe_load(f)
        f.close()
    
    # load exp config
    exp_config_path = glob.glob(os.path.join(args.exp_dir,'*.yaml'))[0]
    with open(exp_config_path) as f:
        exp_config = yaml.safe_load(f)
        f.close()

    # make dir
    os.makedirs(os.path.join(args.exp_dir, 'inference', args.task, args.epochs), exist_ok = True)
    out_wav_dir = os.path.join(args.exp_dir, 'inference', args.task, args.epochs)
    # load eval_list
    with open(args.eval_list ) as f:
        eval_list = json.load(f)
        f.close()
    
    print(f'generating {len(eval_list)} samples')
    # split eval_list by sge_job_idx
    n_jobs_per_task = round(len(eval_list) / args.sge_n_jobs, 0)    
    selected_eval_list = eval_list[int(args.sge_job_idx * n_jobs_per_task): int((args.sge_job_idx+1) * n_jobs_per_task)]
        
    # load encoders
    ling_encoder = exp_config['ling_enc']
    speaker_encoder = exp_config['spk_enc']
    prosodic_encoder = exp_config['pros_enc']
    
    ling_enc_load_func = f'load_{ling_encoder}'
    speaker_enc_load_func = f'load_{speaker_encoder}'
    
    
    ling_enc_model = eval(ling_enc_load_func)(**enc_dec_config[ling_encoder])
    speaker_enc_model = eval(speaker_enc_load_func)(**enc_dec_config[speaker_encoder])
    print(ling_enc_model)
    print(speaker_enc_model)
    # load decoder
    decoder = exp_config['decoder']
    decoder_load_func = f'load_{decoder}'
    
    decoder_model = eval(decoder_load_func)(**enc_dec_config[decoder])
    print(decoder_model)
    # load vocoder
    vocoder = args.vocoder
    vocoder_load_func = f'load_{vocoder}'
    
    vocoder_model = eval(vocoder_load_func)(**enc_dec_config[vocoder])
    print(vocoder_model)

    # conduct inference

    for meta in eval_list:
        # load eval_list metadata
        ID = meta['ID']
        src_wav_path = meta['src_wav']
        trg_wav_path = meta['trg_wav']
        # load src wav & trg wav
        src_wav = load_wav(src_wav, 16000)

        # to tensor
        src_wav_tensor = torch.FloatTensor(src_wav).unsqueeze(0) 



    


    
    




