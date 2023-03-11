from .d_vector.d_vector_model.audio import preprocess_wav
from .d_vector.d_vector_model.voice_encoder import SpeakerEncoder
from speechbrain.pretrained import EncoderClassifier
import torch
import torchaudio


def load_speaker_encoder(spk_enc_type, device = 'cpu'):
    
    if spk_enc_type == 'utt_dvec':
        return load_d_vector( device = device)
    elif spk_enc_type == 'utt_ecapa_tdnn':
        return load_ecapa_tdnn(device = device)
    else:
        return None    
def load_speaker_encoder_func(task, spk_enc_type):
    
    if spk_enc_type == 'utt_dvec':
        if task == 'a2a_vc' or task == 'm2m_vc':
            return d_vector_spk_mean_emb
        elif task == 'oneshot_vc' or 'oneshot_resyn':
            return d_vector_emb 
        raise Exception               
    if spk_enc_type == 'utt_ecapa_tdnn':
        if task == 'a2a_vc' or  'm2m_vc':
            return ecapa_tdnn_spk_mean_emb
        elif task == 'oneshot_vc' or 'oneshot_resyn':
            return ecapa_tdnn_emb 
        raise Exception               
                
        

def load_ecapa_tdnn(ckpt = '', config = None, device = 'cpu'):
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    classifier.eval()
    classifier.to(device)
    return classifier

def load_d_vector(ckpt = 'speaker_encoder/d_vector/d_vector_model/ckpt/pretrained_bak_5805000.pt', config = None, device = 'cpu'):
    
    encoder = SpeakerEncoder(ckpt, device)
    encoder.eval()

    return encoder

def ecapa_tdnn_emb(model, wav_path):
    wav_path = wav_path[0]
    signal, fs =torchaudio.load(wav_path)
    if fs != 16000:
        signal = torchaudio.functional.resample(signal, fs, 16000)

    embeddings = model.encode_batch(signal).squeeze(0).squeeze(0)
    return embedding
    
def ecapa_tdnn_spk_mean_emb(model, wav_paths):
    batch = []
    for wav_path in wav_paths:
        signal, fs =torchaudio.load(wav_path)
        if fs != 16000:
            signal = torchaudio.functional.resample(signal, fs, 16000)
        batch.append(signal)    
    batch = torch.cat(batch, 0)    

    embeddings = model.encode_batch(signal)
    embedding = torch.mean(embeddings, dim = 0).view(-1)
    return embedding

def d_vector_emb(model, wav_path):    
    wav_path = wav_path[0]
    audio = preprocess_wav(wav_path)
    spk_emb = model.embed_utterance(audio)
    return spk_emb

def d_vector_spk_mean_emb(model, wav_paths):
    audios = [preprocess_wav(audio) for audio in wav_paths]

    spk_emb = model.embed_speaker(audios)
    return spk_emb
    
