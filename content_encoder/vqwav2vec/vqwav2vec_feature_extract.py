import torch
import fairseq
import torchaudio
import sys
import numpy as np
import librosa
from scipy.io import wavfile

if len(sys.argv) != 4:
    raise Exception("three arguments needed")
audio_dir=sys.argv[1]
out_dir=sys.argv[2]
scp_dir=sys.argv[3]
import glob
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import subprocess
from tqdm import tqdm

def load_wav(path):
    sr, x = wavfile.read(path)
    signed_int16_max = 2**15
    if x.dtype == np.int16:
        x = x.astype(np.float32) / signed_int16_max
    print(f'48khz wav {x.shape}')
    #x,_ = librosa.effects.trim(x,top_db=60,frame_length=2048,hop_length=512)    
    #print(f'after trim {x.shape}')
    if sr != 16000:
        x = librosa.resample(x, sr, 16000)
    print(f'resample {x.shape}')
    x = np.clip(x, -1.0, 1.0)

    #x,_ = librosa.core.load(path,hparams.sample_rate)
    return x
def process(_audio_paths, audio_dir, out_dir, split):
    cp = 'ckpt/vq-wav2vec_kmeans.pt'
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp])
    model = model[0]
    model.eval()
    for scp, seg in _audio_paths:
        file_id = scp.split(' ')[0]
        speaker = file_id.split('_')[0]
        audio_path = os.path.join(audio_dir, speaker, file_id+'.wav')
        wav = load_wav(audio_path)
        start, end = seg.split(' ')[2], seg.split(' ')[3]
        wav = wav[int(float(start) * 16000): int(float(end) * 16000)]
        window_size=480
        
        wav_input_16khz = np.pad(wav, pad_width = (window_size//2,window_size//2), mode='reflect')
        print(f"after pad {wav_input_16khz.shape}")
        
        wav_input_16khz = torch.FloatTensor(wav_input_16khz).unsqueeze(0)
        z = model.feature_extractor(wav_input_16khz)
        print(f"z {z.size()}")
        dense, idxs = model.vector_quantizer.forward_idx(z)

        dense = dense[0].data.numpy()
        idxs = idxs[0].data.numpy()
        print(f" dense {dense.shape} idxs {idxs.shape}")
        os.makedirs(os.path.join(out_dir,split, speaker), exist_ok = True)
        np.save(os.path.join(out_dir, split, speaker, file_id+'_dense'), dense)
        np.save(os.path.join(out_dir, split, speaker, file_id+'_idxs'), idxs)

splits = ['eval']

#wav_input_16khz = torch.randn(1,10000)
#resampler = torchaudio.transforms.Resample(24000,16000)
for split in splits:
    scp = []
    seg = []
    for spk in ['p360','p361','p362','p363','p364','p374','p376']:
        wav_scp = os.path.join(scp_dir,split+'_'+spk,'wav.scp')
        segment = os.path.join(scp_dir, split+'_'+spk,'segments')
        with open(wav_scp) as f:
            scp_lines = f.readlines()
            scp.extend(scp_lines)
            f.close()
        with open(segment) as f:
            segment_lines = f.readlines()
            seg.extend(segment_lines)
            f.close()
    process(zip(scp, seg), audio_dir, out_dir, split)    
    '''        
    executor = ProcessPoolExecutor(max_workers=20)
    futures = []
    stack = []
    num_stacks = 0
    for scp_line, segment_line in zip(scp_lines, segment_lines):
        if len(stack) <=100:
            stack += [(scp_line,segment_line)]
        else:
            futures.append(executor.submit(partial(process, stack, audio_dir, out_dir, split)))
            stack = []
            num_stacks += 1
            print(f'stack {num_stacks}')
    results = [future.result() for future in tqdm(futures)]    
    '''
