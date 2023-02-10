from .ppg_vc_hifigan.hifigan_model import load_hifigan_generator
from .libritts_hifigan.vocoder import libritts_hifigan_model

def load_ppg_vc_hifigan(ckpt = None, config = None, device = 'cpu'):
    model = load_hifigan_generator(device)
    return model

def load_libritts_hifigan(ckpt = 'vocoder/libritts_hifigan/checkpoint-600000steps.pkl', config = 'vocoder/libritts_hifigan/config.yml', device = 'cpu'):
    
    model = libritts_hifigan_model(ckpt, config, device)
    return model

def libritts_hifigan(model, mel):
    
    wav = model.inference(mel.squeeze(0)).view(-1)        

    return wav

def ppg_vc_hifigan(model, mel):
    
    wav = model(mel.transpose(1,2)).view(-1)
    return wav     
