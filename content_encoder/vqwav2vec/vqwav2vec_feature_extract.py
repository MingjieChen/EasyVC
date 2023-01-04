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

VQWAV2VEC_SAMPLING_RATE=16000
def load_wav(path):
    sr, x = wavfile.read(path)
    signed_int16_max = 2**15
    if x.dtype == np.int16:
        x = x.astype(np.float32) / signed_int16_max
    print(f'original wav {x.shape}')
    if sr != VQWAV2VEC_SAMPLING_RATE:
        x = librosa.resample(x, sr, VQWAV2VEC_SAMPLING_RATE)
    print(f'resample to 16khz {x.shape}')
    x = np.clip(x, -1.0, 1.0)

    return x
def process_speaker(spk_meta, spk, args):
    #cp = 'ckpt/vq-wav2vec_kmeans.pt'
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([args.vqwav2vec_ckpt])
    model = model[0]
    model.eval()
    for row in spk_meta:
        ID = row['ID']
        wav_path = row['wav_path']

        wav = load_wav(wav_path)

        start, end = float(row['start']), float(row['end'])
        wav = wav[int(float(start) * VQWAV2VEC_SAMPLING_RATE): int(float(end) * VQWAV2VEC_SAMPLING_RATE)]
        
        wav_tensor = torch.FloatTensor(wav).unsqueeze(0)
        z = model.feature_extractor(wav_tensor)
        #print(f"z {z.size()}")
        dense, idxs = model.vector_quantizer.forward_idx(z)

        dense = dense[0].data.numpy()
        #idxs = idxs[0].data.numpy()
        print(f" dense {dense.shape} ")
        dump_path=os.path.join(args.dump_dir,args.split, spk, ID+'_vqw2v.npy')
        os.makedirs(os.path.dirname(dump_path), exist_ok = True)
        np.save(dump_path, dense)
        #np.save(os.path.join(out_dir, split, speaker, file_id+'_idxs'), idxs)
    return 0    


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--metadata', type = str)
    parser.add_argument('--vqwav2vec_ckpt' type = str)
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

