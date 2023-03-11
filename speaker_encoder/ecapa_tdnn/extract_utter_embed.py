import torchaudio
from speechbrain.pretrained import EncoderClassifier
import argparse
import csv
import sys
import json
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import subprocess


sampling_rate = 16000
def process_speaker(spk_meta, spk, args):
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

    for row in spk_meta:
        ID = row['ID']
        wav_path = row['wav_path']

        signal, fs =torchaudio.load(wav_path)
        if fs != sampling_rate:
            signal = torchaudio.functional.resample(signal, fs, 16000)

        embeddings = classifier.encode_batch(signal)
        spk_emb = embeddings[0][0].data.numpy()

        print(f'spk_emb {spk_emb.shape}')
        emb_path = os.path.join(args.dump_dir, args.split, 'utt_ecapa_tdnn', spk, f'{ID}.npy')
        os.makedirs(os.path.dirname(emb_path), exist_ok = True)
        np.save(emb_path, spk_emb)
        


    return 0

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--metadata', type = str)
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


