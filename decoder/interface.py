from decoder.fastspeech2.fastspeech2 import FastSpeech2
from decoder.taco_ar.model import Model as TacoAR
from decoder.taco_mol.model import MelDecoderMOLv2 as TacoMOL
import torch
import yaml


def load_fastspeech2(ckpt = None, config = None, device = 'cpu'):
    with open(config) as f:
        model_config = yaml.safe_load(f)
        f.close()
    
        
    model = FastSpeech2(model_config['decoder_params'])
    params = torch.load(ckpt, map_location = torch.device(device))
    params = params['model']

    model.load_state_dict(params)
    model.to(device)
    model.eval()
    return model


def fastspeech2(model, ling, pros, spk):
    
    _, mel, _ = model(ling, pros, spk, torch.LongTensor([ling.size(1)]).to(ling.device), ling.size(1))
    return mel


def load_taco_ar(ckpt = None, config = None, device = 'cpu'):
    with open(config) as f:
        model_config = yaml.safe_load(f)
        f.close()
    
    model = TacoAR(model_config['decoder_params'])
    params = torch.load(ckpt, map_location = torch.device(device))
    params = params['model']

    model.load_state_dict(params)
    model.to(device)
    model.eval()

    return model

 def taco_ar(model, ling, pros, spk):
     
     mel, _ = model(ling, torch.LongTensor([ling.size(1)]).to(ling.device), spk)
     return mel      

def load_taco_mol(ckpt = None, config = None, device = 'cpu'):
    with open(config) as f:
        model_config = yaml.safe_load(f)
        f.close()
    model = TacoMOL(model_config['decoder_params'])
    params = torch.load(ckpt, map_location = torch.device(device))
    params = params['model']

    model.load_state_dict(params)
    model.to(device)
    model.eval()

    return model

def taco_mol(model, ling, pros, spk):
    
    _, mel, _ = model.inference(ling, pros, spk)    
    return mel
        
