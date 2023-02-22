import argparse
import csv
import sys
import json
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import subprocess
from d_vector_model.audio import preprocess_wav
from d_vector_model.voice_encoder import SpeakerEncoder
D_VECTOR_SAMPLING_RATE=16000





def process_speaker(spk_meta, spk, args):
    encoder = SpeakerEncoder(args.d_vector_ckpt, 'cpu')

    for row in spk_meta:
        ID = row['ID']
        wav_path = row['wav_path']
        audio = preprocess_wav(wav_path)
        spk_emb = encoder.embed_utterance(audio)
        print(f'spk_emb {spk_emb.shape}')
        emb_path = os.path.join(args.dump_dir, args.split, 'utt_dvec', spk, f'{ID}.npy')
        os.makedirs(os.path.dirname(emb_path), exist_ok = True)
        np.save(emb_path, spk_emb)
        


    return 0

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--metadata', type = str)
    parser.add_argument('--d_vector_ckpt', type = str)
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


