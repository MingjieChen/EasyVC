from .fastspeech2.fastspeech2 import FastSpeech2
from .taco_ar.model import Model as TacoAR
from .taco_mol.model import MelDecoderMOLv2 as TacoMOL
import torch
import yaml

def remove_module_from_state_dict(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict        

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
    
    _, mel, _ = model.inference(ling, pros, spk)    
    return mel
        
