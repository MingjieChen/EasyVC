import random
import yaml
#from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import sys
#from utils import read_hdf5, write_hdf5
from model import build_model
import argparse
import os
import json
import glob
import soundfile as sf
#from utils import load_wav, logmelspectrogram, to_categorical
from sklearn.preprocessing import StandardScaler
#from data_loader import load_stats
import csv
from tqdm import tqdm
from vocoders.hifigan_model  import load_hifigan_generator
from conformer_ppg_model.build_ppg_model import load_ppg_model
import fairseq
from scipy.io import wavfile
import resampy
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
    parser.add_argument('--out_wav_dir', type = str)
    parser.add_argument('--model_path', type = str)
    parser.add_argument('--in_wav_dir',type = str)
    parser.add_argument('--trg_spk', type = str)
    parser.add_argument('--spk_emb_dir', type = str)
    parser.add_argument('--device', type = str, default = 'cpu')
    args = parser.parse_args()
    print(args)



    # make dir
    os.makedirs(args.out_wav_dir, exist_ok = True)
    # load config
    config_file_path = glob.glob(os.path.join(os.path.dirname(args.model_path),'*.yaml'))[0]
    with open(config_file_path) as f:
        config = yaml.safe_load(f)
    print(config)    
    
    # source wavs 
    source_wavs = glob.glob(os.path.join(args.in_wav_dir,"*.wav")) 
    print(f'got {len(source_wavs)} source wavs')
    # speaker emb
    trg_spk_emb = np.load(os.path.join(args.spk_emb_dir, args.trg_spk+'.npy'))
    
    # build model
    model = build_model(config)
    # load model
    params = torch.load(args.model_path, map_location=torch.device(args.device))    
    params = params['model']
    
    model.generator.load_state_dict(params['generator'])
    model.generator.to(torch.device(args.device))
    model.generator.eval()
    
    
    # build vocoder
    hifigan_model = load_hifigan_generator(args.device)    
    # load scaler
    #scaler = load_stats(config['data_loader']['stats'])

    # build vq extractor
    
    if config['input_type'] == 'vqw2v':
        # build vq extractor
        
        cp = '/share/mini1/res/t/vc/studio/timap-en/vctk/fairseq/examples/wav2vec/ckpt/vq-wav2vec_kmeans.pt'
        vq_model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp])
        vq_model = vq_model[0]
        vq_model.eval()
        vq_model = vq_model.to(args.device)
    elif config['input_type'] == 'ppg':
        ppg_config = '/share/mini1/res/t/vc/studio/timap-en/vctk/model/transformer_adversarial/conformer_ppg_model/en_conformer_ctc_att/config.yaml'
        ppg_ckpt = '/share/mini1/res/t/vc/studio/timap-en/vctk/model/transformer_adversarial/conformer_ppg_model/en_conformer_ctc_att/24epoch.pth'
        ppg_model = load_ppg_model(ppg_config, ppg_ckpt, args.device)    
    
    
    for wav_path in source_wavs:
        
        # load source wav
        wav_input_16khz = load_wav(wav_path)
        # extract vq features
        #window_size=480
        #wav_input_16khz = np.pad(wav, pad_width = (window_size//2,window_size//2), mode='reflect')
        wav_input_16khz = torch.FloatTensor(wav_input_16khz).to(args.device).unsqueeze(0)
        wav_length = torch.LongTensor([wav_input_16khz.size(1)]).to(args.device)
        if config['input_type'] == 'vqw2v':
            z = vq_model.feature_extractor(wav_input_16khz)
            dense, idxs = vq_model.vector_quantizer.forward_idx(z)
            ling_features = dense.transpose(1,2)
            print(f"vq {ling_features.size()}")
        elif config['input_type'] == 'ppg':
            ling_features = ppg_model(wav_input_16khz, wav_length)

        # trg_spk_emb
        y_trg = torch.FloatTensor([trg_spk_emb]).unsqueeze(1).to(args.device)
        length = torch.LongTensor([ling_features.size(1)]).to(args.device)
        
        mel,  _ = model.generator(ling_features, y_trg, length, ling_features.size(1))
        converted_feat = mel    
        print(f"converted_feat {converted_feat.shape}")
        
        converted_wav = hifigan_model(converted_feat.transpose(1,2)).view(-1) 
        #converted_feat = converted_feat.squeeze(0).squeeze(0).data.numpy().T
        #converted_feat = scaler.inverse_transform(converted_feat)
        #out_path = os.path.join(args.out_feat_dir, f'{ID}-feats')
        src_spk =  os.path.basename(wav_path).split('_')[0]
        file_id = os.path.basename(wav_path).split('_')[1].split('.')[0]
        converted_wav_basename = f'{src_spk}_{args.trg_spk}_{file_id}_gen.wav'
        #np.save(out_path+'.npy', converted_feat)    
        sf.write(os.path.join(args.out_wav_dir,converted_wav_basename), converted_wav.data.cpu().numpy(), 24000, "PCM_16")
