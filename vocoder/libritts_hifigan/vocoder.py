import parallel_wavegan
from parallel_wavegan.utils import load_model, read_hdf5
import yaml
import torch
def libritts_hifigan_model(ckpt, config, device):
    with open(config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    device = torch.device(device)
    model = load_model(ckpt, config)
    model.remove_weight_norm()
    model.to(device)    
    return model

