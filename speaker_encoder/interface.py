from d_vector.d_vector_model.audio import preprocess_wav
from d_vector.d_vector_model.voice_encoder import SpeakerEncoder



def load_d_vector(ckpt = None, config = None, device = None):
    
    encoder = SpeakerEncoder(ckpt, device)
    encoder.eval()

    return encoder

def d_vector_emb(model, wav_path):    
    audio = preprocess_wav(wav_path)
    spk_emb = encoder.embed_utterance(audio)
    return spk_emb

def d_vec_spk_mean_emb(model, wav_paths):
    audios = [preprocess_wav(audio) for audio in wav_paths]

    spk_emb = encoder.embed_speaker(audios)
    return spk_emb
    
