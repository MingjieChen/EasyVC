import argparse
import torch
import torchaudio
import sys
import numpy as np
import librosa
from scipy.io import wavfile
import csv

import glob
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import subprocess
from tqdm import tqdm
print(sys.path)
from conformer_ppg_model.build_ppg_model import load_ppg_model

CONFORMER_PPG_SAMPLING_RATE=16000
def load_wav(path):
    sr, x = wavfile.read(path)
    signed_int16_max = 2**15
    if x.dtype == np.int16:
        x = x.astype(np.float32) / signed_int16_max
    print(f'original wav {x.shape}')
    if sr != CONFORMER_PPG_SAMPLING_RATE:
        x = librosa.resample(x, sr, CONFORMER_PPG_SAMPLING_RATE)
    print(f'resample to 16khz {x.shape}')
    x = np.clip(x, -1.0, 1.0)

    return x
def extract_ppg(wav, original_sr, ppg_model):
    
    if original_sr != 16000:
        wav = librosa.resample(wav, original_sr, 16000)
    wav_tensor = torch.FloatTensor([wav])
    wav_length = torch.LongTensor([wav.shape[0]])
    
    with torch.no_grad():
        bnf = ppg_model(wav_tensor, wav_length) 
    bnf_npy = bnf.squeeze(0).cpu().numpy()
    return bnf_npy
def process_speaker(spk_meta, spk, args):
    ppg_model = load_ppg_model(args.conformer_ppg_config, args.conformer_ppg_ckpt, 'cpu')
    ppg_model.eval()
    for row in spk_meta:
        ID = row['ID']
        wav_path = row['wav_path']

        wav = load_wav(wav_path)

        start, end = float(row['start']), float(row['end'])
        wav = wav[int(float(start) * CONFORMER_PPG_SAMPLING_RATE): int(float(end) * CONFORMER_PPG_SAMPLING_RATE)]
        
        #wav_tensor = torch.FloatTensor(wav).unsqueeze(0)
        # extract ppg
        ppg = extract_ppg(wav, CONFORMER_PPG_SAMPLING_RATE, ppg_model)
        ppg_path = os.path.join(args.dump_dir, args.split, spk, ID+'_cppg.npy')
        os.makedirs(os.path.dirname(ppg_path), exist_ok = True)
        np.save(ppg_path, ppg)
    return 0    


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--metadata', type = str)
    parser.add_argument('--conformer_ppg_config', type = str)
    parser.add_argument('--conformer_ppg_ckpt', type = str)
    parser.add_argument('--dump_dir', type = str)
    parser.add_argument('--split', type = str)
    parser.add_argument('--max_workers', type = int, default = 20)
    parser.add_argument('--speaker', type = str, default = None)
    args = parser.parse_args()

    # build a dict for spk2metadata
    spk2meta = {}
    with open(args.metadata) as f:
        reader = csv.DictReader(f)
        for row in reader:
            _spk = row['spk']
            if _spk not in spk2meta:
                spk2meta[_spk] = []
            
            spk2meta[_spk].append(row)         
        
        f.close()
    
    if args.speaker is not None:
        # only for one speaker
        if args.speaker not in spk2meta:
            raise Exception(f"speaker {speaker} should be in the metadata")

        spk_meta = spk2meta[args.speaker]
        process_speaker(spk_meta, args.speaker, args)
    else:
        # process all speakers
        
        # set up processes    
        executor = ProcessPoolExecutor(max_workers=args.max_workers)
        futures = []
        for spk in spk2meta:
            spk_meta = spk2meta[spk]        
        
            futures.append(executor.submit(partial(process_speaker, spk_meta, spk, args)))
        results = [future.result() for future in tqdm(futures)]    

