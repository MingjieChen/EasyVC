from .conformer_ppg.conformer_ppg_model.build_ppg_model import load_ppg_model
import torch

def load_vqwav2vec(ckpt = None, config = None, device = None):
    import fairseq
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt])
    model = model[0]
    model.eval()
    return model

def vqwav2vec(model, wav_tensor):
        
    z = model.feature_extractor(wav_tensor)
    dense, idxs = model.vector_quantizer.forward_idx(z)

    dense = dense[0].data.numpy().T
    return

def load_conformer_ppg(ckpt = None, config = None, device = 'cpu'):
    ppg_model = load_ppg_model(config, ckpt, device)
    ppg_model.eval()
    return ppg_model

def conformer_ppg(model, wav_tensor):
        
    wav_length = torch.LongTensor([wav_tensor.size(1)]).to(wav_tensor.device)
    
    with torch.no_grad():
        bnf = model(wav_tensor, wav_length) 
    bnf_npy = bnf.squeeze(0).cpu().numpy()
    return bnf_npy

def load_hubert_km100(ckpt = None, config = None, device = None):
        
    model = torch.hub.load("bshall/hubert:main", "hubert_discrete")
    model.eval()
    return model

def hubert_km100(model, wav_tensor):
    wav_tensor = wav_tensor.unsqueeze(0)    

    with torch.inference_mode():

       dense = model.units(wav_tensor)


    dense = dense.data.numpy()
    
    return dense

def load_hubert_soft(ckpt = None, config = None, device = None):
    model = torch.hub.load("bshall/hubert:main", "hubert_soft")
    model.eval()
    return model


def hubert_soft(model, wav_tensor):
    wav_tensor = wav_tensor.unsqueeze(0)    

    with torch.inference_mode():

       dense = model.units(wav_tensor)


    dense = dense[0].data.numpy()
    
    return dense
        
        
        
    



