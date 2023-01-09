from d_vector_model.voice_encoder import SpeakerEncoder
import csv
import argparse
import sys
import json
import os
import numpy as np
from d_vector_model.audio import preprocess_wav
D_VECTOR_SAMPLING_RATE=16000

def load_wav(path):
    sr, x = wavfile.read(path)
    signed_int16_max = 2**15
    if x.dtype == np.int16:
        x = x.astype(np.float32) / signed_int16_max
    print(f'original wav {x.shape}')
    if sr != D_VECTOR_SAMPLING_RATE:
        x = librosa.resample(x, sr, D_VECTOR_SAMPLING_RATE)
    print(f'resample to 16khz {x.shape}')
    x = np.clip(x, -1.0, 1.0)

    return x
def process_speaker(spk_meta, spk, args):
    encoder = SpeakerEncoder(args.d_vector_ckpt, 'cpu')

    wav_files = []
    for row in spk_meta:
        wav_path = row['wav_path']
        wav_files.append(wav_path)

    print(f'obtained {len(wav_files)} for spk {spk}')        
    # loop through speakers
    audios = [preprocess_wav(audio) for audio in wav_files]
    spk_emb = encoder.embed_speaker(audios)
    print(f'spk_emb {spk} {spk_emb.shape}')
    emb_path = os.path.join(args.dump_dir, args.split, 'spk_dvec', f'{spk}.npy')
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


