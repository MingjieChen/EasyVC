import argparse
import yaml
import json
import os
import glob
from audio_utils import mel_spectrogram, normalize
import pyworld as pw
import librosa
import numpy  as np
from tqdm import tqdm
import csv
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import subprocess
import torch



def process_speaker(spk_meta, spk, config, args):
    
    for row in spk_meta:
        # load wav
        ID = row['ID']
        wav_path = row['wav_path'].strip()
        audio, fs = librosa.load(wav_path, sr = config['sampling_rate'])
        
        # trim silence
        start, end = float(row['start']), float(row['end'])
        audio = audio[ int(start * config['sampling_rate']):
                        int(end * config['sampling_rate'])
            ]
        
        # extract mel   
        norm_audio = normalize(audio) * 0.95
        
        norm_audio = torch.FloatTensor(norm_audio).unsqueeze(0)
        mel = mel_spectrogram(
               norm_audio,
               sampling_rate = config['sampling_rate'],
               n_fft = config['fft_size'],
               hop_size = config['hop_size'],
               win_size = config['win_length'],
               num_mels = config['num_mels'],
               fmin = config['fmin'],
               fmax = config['fmax'],
        )
        mel = mel.squeeze(0).T.numpy()
        # min-max normalization
        mel = (mel - config['mel_min']) / (config['mel_max'] - config['mel_min']) * 8.0 - 4.0 
        mel = np.clip(mel, -4. , 4.)
        mel_path = os.path.join(args.dump_dir, args.split, 'mel', spk, ID+'.npy')
        os.makedirs(os.path.dirname(mel_path), exist_ok = True)
        np.save(mel_path, mel)
        # extract pitch
        pitch, t = pw.harvest(
            audio.astype(np.float64),
            config['sampling_rate'],
            frame_period=config['hop_size'] / config['sampling_rate'] * 1000,
            f0_floor = config['f0_floor'],
            f0_ceil = config['f0_ceil']
        )
        pitch = pitch.astype(np.float32)
        pitch_path = os.path.join(args.dump_dir, args.split, 'f0', spk, ID+'.npy')
        os.makedirs(os.path.dirname(pitch_path), exist_ok = True)
        np.save(pitch_path, pitch )
    return 0    
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--metadata', type = str)
    parser.add_argument('--config_path', type = str)
    parser.add_argument('--dump_dir', type = str)
    parser.add_argument('--split', type = str)
    parser.add_argument('--max_workers', type = int, default = 20)
    parser.add_argument('--speaker', type = str, default = None)
    args = parser.parse_args()
    
    # load in config
    with open(args.config_path) as f:
        config = yaml.safe_load(f)
        f.close()
    print(config)   
    
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
        process_speaker(spk_meta, args.speaker, config, args)
    
    else:
        # process all speakers
        
        # set up processes    
        executor = ProcessPoolExecutor(max_workers=args.max_workers)
        futures = []
        for spk in spk2meta:
            spk_meta = spk2meta[spk]        
        
            futures.append(executor.submit(partial(process_speaker, spk_meta, spk, config, args)))
        results = [future.result() for future in tqdm(futures)]    
