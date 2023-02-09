from .d_vector.d_vector_model.audio import preprocess_wav
from .d_vector.d_vector_model.voice_encoder import SpeakerEncoder


def load_speaker_encoder(spk_enc_type, device = 'cpu'):
    
    if spk_enc_type == 'utt_dvec':
        return load_d_vector( device = device)
    else:
        return None    
def load_speaker_encoder_func(task, spk_enc_type):
    
    if spk_enc_type == 'utt_dvec':
        if task == 'a2a_vc':
            return d_vector_spk_mean_emb
        elif task == 'oneshot_vc' or 'oneshot_resyn':
            return d_vector_emb            
        


def load_d_vector(ckpt = 'speaker_encoder/d_vector/d_vector_model/ckpt/pretrained_bak_5805000.pt', config = None, device = 'cpu'):
    
    encoder = SpeakerEncoder(ckpt, device)
    encoder.eval()

    return encoder

def d_vector_emb(model, wav_path):    
    wav_path = wav_path[0]
    audio = preprocess_wav(wav_path)
    spk_emb = model.embed_utterance(audio)
    return spk_emb

def d_vector_spk_mean_emb(model, wav_paths):
    audios = [preprocess_wav(audio) for audio in wav_paths]

    spk_emb = model.embed_speaker(audios)
    return spk_emb
    
