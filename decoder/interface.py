from .fastspeech2.fastspeech2 import FastSpeech2
from .taco_ar.model import Model as TacoAR
from .taco_mol.model import MelDecoderMOLv2 as TacoMOL
from .vits.models import VITS
from .grad_tts.grad_tts_model import GradTTS
from .diffwave.model import DiffWave
import numpy as np
import torch
import yaml

def remove_module_from_state_dict(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v    
    return new_state_dict        



def load_DiffWave(ckpt = None, config = None, device = 'cpu'):
    with open(config) as f:
        model_config = yaml.safe_load(f)
        f.close()
    
    model = DiffWave(model_config['decoder_params'])    
    params = torch.load(ckpt, map_location = torch.device(device))
    params = params['model']
    params = remove_module_from_state_dict(params)
    model.load_state_dict(params)

    model.eval()
    model.to(device)
    return model

    

def load_VITS(ckpt = None, config = None, device = 'cpu'):
    with open(config) as f:
        model_config = yaml.safe_load(f)
        f.close()
    model = VITS(model_config['decoder_params'])    
    params = torch.load(ckpt, map_location = torch.device(device))
    params = params['model']
    params = remove_module_from_state_dict(params)
    model.load_state_dict(params)

    generator = model.generator
    generator.dec.remove_weight_norm()
    generator.to(device)
    generator.eval()
    return generator
    
    
def load_GradTTS(ckpt = None, config = None, device = 'cpu'):
    with open(config) as f:
        model_config = yaml.safe_load(f)
        f.close()
    
        
    model = GradTTS(model_config['decoder_params'])
    params = torch.load(ckpt, map_location = torch.device(device))
    params = params['model']
    params = remove_module_from_state_dict(params)

    model.load_state_dict(params)
    model.to(device)
    model.eval()
    return model
    

def load_FastSpeech2(ckpt = None, config = None, device = 'cpu'):
    with open(config) as f:
        model_config = yaml.safe_load(f)
        f.close()
    
        
    model = FastSpeech2(model_config['decoder_params'])
    params = torch.load(ckpt, map_location = torch.device(device))
    params = params['model']
    params = remove_module_from_state_dict(params)

    model.load_state_dict(params)
    model.to(device)
    model.eval()
    return model


def infer_FastSpeech2(model, ling, pros, spk):
    
    mel, postnet_mel, _ = model(ling, pros, spk, torch.LongTensor([ling.size(1)]).to(ling.device), ling.size(1))

    if postnet_mel is not None:
        return postnet_mel
    else:    
        return mel


def load_TacoAR(ckpt = None, config = None, device = 'cpu'):
    with open(config) as f:
        model_config = yaml.safe_load(f)
        f.close()
    
    model = TacoAR(model_config['decoder_params'])
    params = torch.load(ckpt, map_location = torch.device(device))
    params = params['model']
    params = remove_module_from_state_dict(params)

    model.load_state_dict(params)
    model.to(device)
    model.eval()

    return model


def infer_VITS(model, ling, pros, spk):
    ling_lengths = torch.LongTensor([ling.size(1)]).to(ling.device)
    ling = ling.transpose(1,2)
    pros = pros.transpose(1,2)
    spk = spk.transpose(1,2)
    out = model.infer(ling, ling_lengths, pros, spk)
    return out

def infer_TacoAR(model, ling, pros, spk):
     
     mel, _ = model(ling, torch.LongTensor([ling.size(1)]).to(ling.device), spk, pros_rep = pros)
     return mel      

def load_TacoMOL(ckpt = None, config = None, device = 'cpu'):
    with open(config) as f:
        model_config = yaml.safe_load(f)
        f.close()
    model = TacoMOL(model_config['decoder_params'])
    params = torch.load(ckpt, map_location = torch.device(device))
    params = params['model']
    params = remove_module_from_state_dict(params)

    model.load_state_dict(params)
    model.to(device)
    model.eval()

    return model

def infer_TacoMOL(model, ling, pros, spk):
    
    mel = model.inference(ling, pros, spk)    
    return mel

def infer_GradTTS(model, ling, pros, spk):
    ling = ling.transpose(1,2)
    pros = pros.transpose(1,2)
    if ling.size(2) %4 != 0:
        pad_length = 4- ling.size(2) % 4
        ling = torch.nn.functional.pad(ling, [0, pad_length])
        pros = torch.nn.functional.pad(pros, [0, pad_length])
    
    
    ling_lengths = torch.LongTensor([ling.size(2)]).to(ling.device)
    mel = model(ling, ling_lengths, spk, pros, 10)        
    mel = mel.transpose(1,2)
    return mel

def infer_DiffWave(model, ling, pros, spk):
    print(f'ling length {ling.size(1)}', flush = True)
    ling_lengths = torch.LongTensor([ling.size(1)]).to(ling.device)
    ling = ling.transpose(1,2)
    pros = pros.transpose(1,2)
    spk = spk.transpose(1,2)
    audio = model.inference(ling, pros, spk, ling_lengths)
    return audio
