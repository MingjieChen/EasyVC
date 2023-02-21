import time
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
from sklearn.preprocessing import StandardScaler



def denorm_mel(mean_tensor, std_tensor, mel):
    
    if mean_tensor is not None and std_tensor is not None:
        mean_tensor = torch.FloatTensor(scaler.mean_)
        std_tensor = torch.FloatTensor(scaler.scale_)
        
        mel = mel * std_tensor + mean_tensor
    
    return mel

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
    #exp 
    parser.add_argument('--epochs', type = str)
    parser.add_argument('--task', type = str) 
    # vocoder
    parser.add_argument('--vocoder', type = str, default = 'ppg_vc_hifigan')
    # sge task 
    parser.add_argument('--sge_task_id', type = int, default = 1)
    parser.add_argument('--sge_n_tasks', type = int, default = 1)

    # arguments
    args = parser.parse_args()
    print(args)

    
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
    n_per_task = np.ceil(len(eval_list) / args.sge_n_tasks)    
    start = int(( args.sge_task_id -1 ) * n_per_task)
    if int( args.sge_task_id * n_per_task) >= len(eval_list):
        end = len(eval_list) -1
    else:
        end = int(args.sge_task_id  * n_per_task)    
    print(f'selected_eval_list from {start} to {end}')    
    selected_eval_list = eval_list[start: end]
        
    # encoders types
    ling_encoder = exp_config['ling_enc']
    speaker_encoder = exp_config['spk_enc']
    prosodic_encoder = exp_config['pros_enc']
    
    # load ling_encoder
    ling_enc_load_func = f'load_{ling_encoder}'
    ling_enc_model = eval(ling_enc_load_func)(device = args.device)
    ling_encoder_func = f'{ling_encoder}'
    # load speaker encoder
    speaker_enc_model = load_speaker_encoder(speaker_encoder, device = args.device)
    speaker_encoder_func = load_speaker_encoder_func(args.task, speaker_encoder)
    print(f'load ling_encoder {ling_encoder} done')
    print(f'load speaker_encoder {speaker_encoder} done')
    # load decoder
    decoder = exp_config['decoder']
    decoder_load_func = f'load_{decoder}'
    decoder_func = f'infer_{decoder}'
    decoder_model_path = os.path.join(args.exp_dir, 'ckpt', f'epoch_{args.epochs}.pth')
    decoder_model = eval(decoder_load_func)(decoder_model_path, exp_config_path, device = args.device)
    print(f'load decoder {decoder} done')
    # load vocoder
    vocoder = args.vocoder
    vocoder_load_func = f'load_{vocoder}'
    vocoder_model = eval(vocoder_load_func)(device = args.device)
    vocoder_func = f'{vocoder}'
    print(f'load vocoder {vocoder} done')
    # conduct inference
    
    # denorm mel scaler
    if 'mel_stats' in exp_config:
        scaler = StandardScaler()
        scaler.mean_ = np.load(exp_config['mel_stats'])[0]
        scaler.scale_ = np.load(exp_config['mel_stats'])[1]
        scaler.n_features_in = scaler.mean_.shape[0]
        mean_tensor = torch.FloatTensor(scaler.mean_).to(args.device)
        std_tensor = torch.FloatTensor(scaler.scale_).to(args.device)
    else:
        mean_tensor = None
        std_tensor = None    

    total_rtf = 0.0
    cnt = 0
    for meta in tqdm(selected_eval_list):
        # load eval_list metadata
        ID = meta['ID']
        src_wav_path = meta['src_wav']
        trg_wav_path = meta['trg_wav']
        # load src wav & trg wav
        src_wav = load_wav(src_wav_path, 16000)
        
        # to tensor
        src_wav_tensor = torch.FloatTensor(src_wav).unsqueeze(0).to(args.device) 
        start_time = time.time()
        # extract ling representations
        ling_rep = eval(ling_encoder_func)(ling_enc_model, src_wav_tensor)
        # trg spk emb
        spk_emb = speaker_encoder_func(speaker_enc_model, trg_wav_path)
        spk_emb_tensor = torch.FloatTensor(spk_emb).unsqueeze(0).unsqueeze(0).to(args.device)
        # generate mel
        mel = eval(decoder_func)(decoder_model, ling_rep, None, spk_emb_tensor)
        mel = denorm_mel(mean_tensor, std_tensor, mel)
        # vocoder
        wav = eval(vocoder_func)(vocoder_model, mel)
        end_time = time.time()
        rtf = (end_time - start_time) / (0.01 * ling_rep.size(1))
        total_rtf += rtf
        cnt += 1
        converted_wav_basename = f'{ID}_gen.wav'
        sf.write(os.path.join(out_wav_dir, converted_wav_basename), wav.data.cpu().numpy(), 24000, "PCM_16")
    print(f"RTF: {total_rtf/cnt :.2f}")    





    


    
    




