from .conformer_ppg.conformer_ppg_model.build_ppg_model import load_ppg_model
from .whisper_ppg.whisper_ppg_model.audio import  pad_or_trim as whisper_ppg_pad_or_trim, log_mel_spectrogram as whisper_ppg_log_mel_spectrogram
from .whisper_ppg.whisper_ppg_model.model import Whisper, ModelDimensions
import torch


def load_whisper_ppg_small(ckpt = 'ling_encoder/whisper_ppg/ckpt/small.pt', config = None, device = 'cpu'):
    checkpoint = torch.load(ckpt, map_location=device)
    dims = ModelDimensions(**checkpoint["dims"])
    model = Whisper(dims)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model.to(device)
    

def load_contentvec_500(ckpt = 'ling_encoder/contentvec_500/contentvec_500_model.pt', config = None, device = 'cpu'):
    import fairseq
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt])
    model = model[0]
    model.eval()
    return model
def load_contentvec_100(ckpt = 'ling_encoder/contentvec_100/contentvec_100_model.pt', config = None, device = 'cpu'):
    import fairseq
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt])
    model = model[0]
    model.eval()
    return model

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

def contentvec_500(model, wav_tensor):
        
    dense = model.feature_extractor(wav_tensor)

    return dense.transpose(1,2)    
def contentvec_100(model, wav_tensor):
        
    dense = model.feature_extractor(wav_tensor)

    return dense.transpose(1,2)    
        
def whisper_ppg_small(model, wav_tensor):
    wav_tensor = wav_tensor.view(-1)
    wav_len = wav_tensor.size(0)
    ppg_len = wav_len // 320
    wav_tensor = whisper_ppg_pad_or_trim(wav_tensor)
    mel = whisper_ppg_log_mel_spectrogram(wav_tensor).to(model.device)
    
    ppg = model.encoder(mel.unsqueeze(0))
    ppg = ppg[:,:ppg_len,:]
    return ppg
            
    



