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
import logging
import logging
logger = logging.getLogger('numba')
logger.setLevel(logging.WARNING)
from ling_encoder.interface import *
from speaker_encoder.interface import *
from prosodic_encoder.interface import *
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
    parser.add_argument('--src_resyn', default = False, action = 'store_true')
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
        end = len(eval_list) 
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
    if 'vocoder' in exp_config:
        vocoder = exp_config['vocoder']
        vocoder_load_func = f'load_{vocoder}'
        vocoder_model = eval(vocoder_load_func)(device = args.device)
        vocoder_func = f'{vocoder}'
        print(f'load vocoder {vocoder} done')
    else:
        vocoder = None
        vocoder_load_func = None
        vocoder_model = None
        vocoder_func = None    
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
    
    # norm pros reps
    if 'pros_stats' in exp_config:
        pros_stats = exp_config['pros_stats']
    else:
        pros_stats = None    

    total_rtf = 0.0
    cnt = 0
    for meta in tqdm(selected_eval_list):
        # load eval_list metadata
        ID = meta['ID']
        src_wav_path = meta['src_wav']
        trg_wav_path = meta['trg_wav']
        
        if args.src_resyn and vocoder == 'ppgvc_hifigan':
            from feature_extraction import ppgvc_hifigan_logmelspectrogram
            src_audio = load_wav(src_wav_path, 24000)
            ppgvc_mel_config = {'sampling_rate':24000, 
                                'fft_size': 1024, 
                                'hop_size': 240,
                                'win_length': 1024,
                                'window': 'hann',
                                'num_mels': 80,
                                'fmin': 0,
                                'fmax': 8000,
                                'mel_min': -12.0,
                                'mel_max': 2.5
                                }
            src_mel_resyn = ppgvc_hifigan_logmelspectrogram(src_audio,ppgvc_mel_config)
        
        # load src wav & trg wav
        src_wav = load_wav(src_wav_path, 16000)
        mel_duration = len(src_wav) // 160 # estimate a mel duration for pad ling and pros reps
        
        # to tensor
        src_wav_tensor = torch.FloatTensor(src_wav).unsqueeze(0).to(args.device) 
        start_time = time.time()
        # extract ling representations
        ling_rep = eval(ling_encoder_func)(ling_enc_model, src_wav_tensor)
        ling_duration = ling_rep.size(1)
        # check if need upsample ling rep
        factor = int(round(mel_duration / ling_duration))
        if factor > 1:
            ling_rep = torch.repeat_interleave(ling_rep, repeats=factor, dim=1)
            ling_duration = ling_rep.size(1)
        if ling_duration > mel_duration:
            ling_rep = ling_rep[:, :mel_duration, :]
        elif mel_duration > ling_duration:
            pad_vec = ling_rep[:, -1, :]
            ling_rep = torch.cat([ling_rep, pad_vec.unsqueeze(1).expand(1, mel_duration - ling_duration, ling_rep.size(2))], dim = 1)
            
        # extract prosodic representations
        if prosodic_encoder != 'none':
            prosodic_func = f'infer_{prosodic_encoder}'
            pros_rep = eval(prosodic_func)(src_wav_path, trg_wav_path, stats = pros_stats)
            pros_duration = pros_rep.size(1)
            if pros_duration > mel_duration:
                pros_rep = pros_rep[:, : mel_duration, :]
            elif mel_duration > pros_duration:
                pad_vec = pros_rep[:, -1, :]
                pros_rep = torch.cat([pros_rep, pad_vec.unsqueeze(1).expand(1, mel_duration - pros_duration, pros_rep.size(2))], dim = 1)
        else:
            pros_rep = None    
        # trg spk emb
        spk_emb = speaker_encoder_func(speaker_enc_model, trg_wav_path)
        spk_emb_tensor = torch.FloatTensor(spk_emb).unsqueeze(0).unsqueeze(0).to(args.device)

        # generate mel
        decoder_out = eval(decoder_func)(decoder_model, ling_rep, pros_rep, spk_emb_tensor)
        decoder_out = denorm_mel(mean_tensor, std_tensor, decoder_out)
        
        if vocoder is not None:
            # vocoder
            wav = eval(vocoder_func)(vocoder_model, decoder_out)
            if args.src_resyn:
                src_mel_tensor = torch.FloatTensor([src_mel_resyn])
                src_resyn_wav = eval(vocoder_func)(vocoder_model, src_mel_tensor)
        else:
            wav = decoder_out.view(-1)    
        end_time = time.time()
        rtf = (end_time - start_time) / (0.01 * ling_rep.size(1))
        total_rtf += rtf
        cnt += 1
        converted_wav_basename = f'{ID}_gen.wav'
        sf.write(os.path.join(out_wav_dir, converted_wav_basename), wav.data.cpu().numpy(), 24000, "PCM_16")
        if args.src_resyn:
            resyn_wav_basename = f'{ID}_resyn.wav'
            sf.write(os.path.join(out_wav_dir, resyn_wav_basename), src_resyn_wav.data.cpu().numpy(), 24000, "PCM_16")

    print(f"RTF: {total_rtf/cnt :.2f}")    





    


    
    




