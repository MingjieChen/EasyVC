from .ppg_vc_hifigan.hifigan_model import load_hifigan_generator
from .libritts_hifigan.vocoder import libritts_hifigan_model
from .vctk_hifigan.vocoder import vctk_hifigan_model
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch

def load_ppgvc_hifigan(ckpt = None, config = None, device = 'cpu'):
    model = load_hifigan_generator(device)
    return model

def load_libritts_hifigan(ckpt = 'vocoder/libritts_hifigan/model.pkl', config = 'vocoder/libritts_hifigan/config.yml', stats = 'vocoder/libritts_hifigan/stats.npy', device = 'cpu'):
    scaler = StandardScaler()
    scaler.mean_ = np.load(stats)[0]
    scaler.scale_ = np.load(stats)[1]
    scaler.n_features_in = scaler.mean_.shape[0]
    

    model = libritts_hifigan_model(ckpt, config, device)
    return (model, scaler)

def libritts_hifigan(model, mel):
    
    hifigan_model, scaler = model
    mean_tensor = torch.FloatTensor(scaler.mean_).to(mel.device)
    std_tensor = torch.FloatTensor(scaler.scale_).to(mel.device)
    mel = (mel - mean_tensor) / (std_tensor + 1e-8)
    wav = hifigan_model.inference(mel.squeeze(0)).view(-1)        

    return wav

def load_vctk_hifigan(ckpt = 'vocoder/vctk_hifigan/model.pkl', config = 'vocoder/vctk_hifigan/config.yml', stats = 'vocoder/vctk_hifigan/stats.npy', device = 'cpu'):
    
    scaler = StandardScaler()
    scaler.mean_ = np.load(stats)[0]
    scaler.scale_ = np.load(stats)[1]
    scaler.n_features_in = scaler.mean_.shape[0]
    model = vctk_hifigan_model(ckpt, config, device)
    return (model, scaler)

def vctk_hifigan(model, mel ):
    

    hifigan_model, scaler = model
    mean_tensor = torch.FloatTensor(scaler.mean_).to(mel.device)
    std_tensor = torch.FloatTensor(scaler.scale_).to(mel.device)
    mel = (mel - mean_tensor) / (std_tensor + 1e-8)
    wav = hifigan_model.inference(mel.squeeze(0)).view(-1)        

    return wav
def ppgvc_hifigan(model, mel):
    
    wav = model(mel.transpose(1,2)).view(-1)
    return wav     
