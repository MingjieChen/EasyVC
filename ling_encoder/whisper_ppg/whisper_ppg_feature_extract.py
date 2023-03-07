import argparse
import csv
import torch
import fairseq
import torchaudio
import sys
import numpy as np
import librosa
from scipy.io import wavfile
import glob
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import subprocess
from tqdm import tqdm

from whisper_ppg_model.audio import load_audio, pad_or_trim, log_mel_spectrogram
from whisper_ppg_model.model import Whisper, ModelDimensions
from whisper_ppg_model.decoding import DecodingOptions, decode

def load_model(path) -> Whisper:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(path, map_location=device)
    dims = ModelDimensions(**checkpoint["dims"])
    print(checkpoint['dims'])
    model = Whisper(dims)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)

SAMPLING_RATE=16000
def process_speaker(spk_meta, spk, args):
    model = load_model(args.ckpt)
    for row in tqdm(spk_meta, total = len(spk_meta)):
        ID = row['ID']
        wav_path = row['wav_path']
        
        wav = load_audio(wav_path)
        start, end = float(row['start']), float(row['end'])
        wav = wav[int(float(start) * SAMPLING_RATE): int(float(end) * SAMPLING_RATE)]
        wav_len = wav.shape[0]
        ppgln = wav_len // 320
        wav = pad_or_trim(wav)
        mel = log_mel_spectrogram(wav).to(model.device)
        
        ppg = model.encoder(mel.unsqueeze(0)).squeeze().data.cpu().float().numpy()
        ppg = ppg[:ppgln,:]
        #print(f"z {z.size()}")

        #idxs = idxs[0].data.numpy()
        print(f" ppg {ppg.shape} ")
        dump_path=os.path.join(args.dump_dir,args.split, f'whisper_ppg_{args.ext}', spk, ID+'.npy')
        os.makedirs(os.path.dirname(dump_path), exist_ok = True)
        np.save(dump_path, ppg)
        #np.save(os.path.join(out_dir, split, speaker, file_id+'_idxs'), idxs)
    return 0    


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--metadata', type = str)
    parser.add_argument('--ckpt', type = str)
    parser.add_argument('--dump_dir', type = str)
    parser.add_argument('--split', type = str)
    parser.add_argument('--max_workers', type = int, default = 20)
    parser.add_argument('--speaker', type = str, default = None)
    parser.add_argument('--ext', type = str, default = 'largev2')
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

