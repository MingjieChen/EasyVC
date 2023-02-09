from .conformer_ppg.conformer_ppg_model.build_ppg_model import load_ppg_model
import torch

def load_vqwav2vec(ckpt = 'ling_encoder/vqwav2vec/vq-wav2vec_kmeans.pt' , config = None, device = 'cpu'):
    import fairseq
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt])
    model = model[0]
    model.to(device)
    model.eval()

    return model

def vqwav2vec(model, wav_tensor):
        
    z = model.feature_extractor(wav_tensor)
    dense, idxs = model.vector_quantizer.forward_idx(z)
    dense = dense.transpose(1,2)
    return dense

def load_conformer_ppg(ckpt = 'ling_encoder/conformer_ppg/conformer_ppg_model/en_conformer_ctc_att/24epoch.pth', config = 'ling_encoder/conformer_ppg/conformer_ppg_model/en_conformer_ctc_att/config.yaml', device = 'cpu'):
    ppg_model = load_ppg_model(config, ckpt, device)
    ppg_model.eval()
    return ppg_model

def conformer_ppg(model, wav_tensor):
        
    wav_length = torch.LongTensor([wav_tensor.size(1)]).to(wav_tensor.device)
    
    with torch.no_grad():
        bnf = model(wav_tensor, wav_length) 
    return bnf

def load_hubert_km100(ckpt = None, config = None, device = 'cpu'):
        
    model = torch.hub.load("bshall/hubert:main", "hubert_discrete")
    model.to(device)
    model.eval()
    return model

def hubert_km100(model, wav_tensor):
    wav_tensor = wav_tensor.unsqueeze(0)    

    with torch.inference_mode():

       dense = model.units(wav_tensor)


    
    return dense

def load_hubert_soft(ckpt = None, config = None, device = 'cpu'):
    model = torch.hub.load("bshall/hubert:main", "hubert_soft")
    model.to(device)
    model.eval()
    return model


def hubert_soft(model, wav_tensor):
    wav_tensor = wav_tensor.unsqueeze(0)    

    with torch.inference_mode():

       dense = model.units(wav_tensor)


    
    return dense
        
        
        
    



