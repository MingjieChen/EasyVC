from .ppg_vc_hifigan.hifigan_model import load_hifigan_generator


def load_ppg_vc_hifigan(ckpt = None, config = None, device = 'cpu'):
    model = load_hifigan_generator(device)
    return model
    

def ppg_vc_hifigan(model, mel):
    
    wav = model(mel.transpose(1,2)).view(-1)
    return wav     
