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
    parser.add_argument('--out_feat_dir', type = str)
    parser.add_argument('--out_wav_dir', type = str)
    parser.add_argument('--model_path', type = str)
    parser.add_argument('--eval_list',type = str)
    parser.add_argument('--out_eval_list', type = str)
    parser.add_argument('--speakers', type = str, default = '../../speaker.json')
    parser.add_argument('--device', type = str, default = 'cpu')
    parser.add_argument('--iters', type = int)
    args = parser.parse_args()
    
    # make dir
    os.makedirs(args.out_wav_dir, exist_ok = True)
    # load config
    config_file_path = glob.glob(os.path.join(os.path.dirname(args.model_path),'*.yaml'))[0]
    with open(config_file_path) as f:
        config = yaml.safe_load(f)
    print(config)    
    # speaker list
    speakers_f = open(config['speakers'])
    speakers = json.load(speakers_f)
    
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
    
    # source feats files
    
    with open(args.eval_list) as csv_f, open( os.path.join(os.path.dirname(args.out_feat_dir), args.out_eval_list),'w') as out_f:
        csv_reader = csv.DictReader(csv_f, delimiter = ',', quotechar = '"')
        csvwriter = csv.writer(out_f, delimiter = ',', quotechar = '"')
        
        csvwriter.writerow(['ID','duration','wav','spk_id','wrd'])
        for row in tqdm(csv_reader):
            wav_path = row['wav']
            ID = row['ID']
            src_spk = ID.split('_')[0]
            trg_spk = row['spk_id']
            # extract mel features
            if not os.path.exists(wav_path):
                raise Exception
            
            # load wav
            
            wav_input_16khz = load_wav(wav_path)
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
                
            
            if 'use_spk_embs' in config['generator'] and config['generator']['use_spk_embs']:
                if 'spk_emb' in row:
                    spk_emb = np.load(row['spk_emb'])   
                else:
                    #spk_emb = np.load(os.path.join(config['speaker_embs'], trg_spk,trg_spk+'_023.npy'))    
                    spk_emb_path = sorted(glob.glob(os.path.join(config['speaker_embs'],trg_spk,'*.npy')))[-1]
                    print(f'spk_emb_path {spk_emb_path}')
                    spk_emb = np.load(spk_emb_path)
                y_trg = torch.FloatTensor([spk_emb]).unsqueeze(1).to(args.device)
            else:    
                trg_id = speakers.index(row['spk_id'])
                src_id = speakers.index(src_spk)
                y_trg = torch.LongTensor([trg_id]).unsqueeze(0).to(args.device)
            length = torch.LongTensor([ling_features.size(1)]).to(args.device)
            
            mel,  _ = model.generator(ling_features, y_trg, length, ling_features.size(1))
            converted_feat = mel    
            print(f"converted_feat {converted_feat.shape}")
            
            converted_wav = hifigan_model(converted_feat.transpose(1,2)).view(-1) 
            #converted_feat = converted_feat.squeeze(0).squeeze(0).data.numpy().T
            #converted_feat = scaler.inverse_transform(converted_feat)
            #out_path = os.path.join(args.out_feat_dir, f'{ID}-feats')
            converted_wav_basename = f'{ID}_gen.wav'
            #np.save(out_path+'.npy', converted_feat)    
            sf.write(os.path.join(args.out_wav_dir,converted_wav_basename), converted_wav.data.cpu().numpy(), 24000, "PCM_16")
            csvwriter.writerow([row['ID'],row['duration'],os.path.join(args.out_wav_dir,converted_wav_basename),row['spk_id'],row['wrd']])




