
from ling_encoder.interface import *
from speaker_encoder.interface import *
from prosodic_encoder.interface import *
from decoder.interface import *
from vocoder.interface import *
import yaml
import resampy
import numpy as np
import soundfile as sf
import os
import sys
import argparse




   

def load_audio(audio_path, sample_rate = 16000):
    audio, sr = librosa.load(audio_path, sample_rate)
    #audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio.transpose(1, 0))
    if sr != sample_rate:
        audio = resampy.resample(audio, sr, sample_rate)
    audio = np.clip(audio, -1.0, 1.0)
    return audio

def load_model(args):

    
    dec_path = args.model_ckpt
    config_path = args.model_config

    with open(config_path) as f:
        exp_config = yaml.safe_load(f)
        f.close()
    
    # encoders types
    ling_encoder = exp_config['ling_enc']
    speaker_encoder = exp_config['spk_enc']
    prosodic_encoder = exp_config['pros_enc']
    # load ling_encoder
    ling_enc_load_func = f'load_{ling_encoder}'
    ling_enc = eval(ling_enc_load_func)(device = args.device)
    ling_enc_func = f'{ling_encoder}'
    # pros_enc
    pros_enc_func = f'infer_{prosodic_encoder}'
    # load speaker encoder
    spk_enc = load_speaker_encoder(speaker_encoder, device = args.device)
    spk_enc_func = load_speaker_encoder_func('oneshot_vc', speaker_encoder)
    # load decoder
    decoder = exp_config['decoder']
    decoder_load_func = f'load_{decoder}'
    dec_func = f'infer_{decoder}'
    dec = eval(decoder_load_func)(dec_path, config_path, device = args.device)

    if 'vocoder' in exp_config:
        vocoder = exp_config['vocoder']
        vocoder_load_func = f'load_{vocoder}'
        voc = eval(vocoder_load_func)(device = args.device)
        voc_func = f'{vocoder}'
    else:
        voc_func = None    
        voc = None
    return ling_enc_func, ling_enc, spk_enc_func, spk_enc, pros_enc_func, dec_func, dec, voc_func, voc
        




    

    
        


def vc_fn(args, ling_enc_func, ling_enc, spk_enc_func, spk_enc, pros_enc_func, dec_func, dec, voc_func, voc):
    
    
    
    src_audio = load_audio(args.source_wav, 16000)    
    print(f'load src_audio done')
    mel_duration = len(src_audio)//160
    
    # extract ling reps
    src_wav_tensor = torch.FloatTensor(src_audio).unsqueeze(0).to(args.device)
    ling_rep = eval(ling_enc_func)(ling_enc, src_wav_tensor)
    ling_duration = ling_rep.size(1)
    factor = int(round(mel_duration / ling_duration))
    if factor > 1:
        ling_rep = torch.repeat_interleave(ling_rep, repeats=factor, dim=1)
        ling_duration = ling_rep.size(1)
    if ling_duration > mel_duration:
        ling_rep = ling_rep[:, :mel_duration, :]
    elif mel_duration > ling_duration:
        pad_vec = ling_rep[:, -1, :]
        ling_rep = torch.cat([ling_rep, pad_vec.unsqueeze(1).expand(1, mel_duration - ling_duration, ling_rep.size(2))], dim = 1)

    # extract pros reps
    pros_rep = eval(pros_enc_func)(args.source_wav, args.target_wav_list).to(args.device)
    pros_duration = pros_rep.size(1)
    if pros_duration > mel_duration:
        pros_rep = pros_rep[:, : mel_duration, :]
    elif mel_duration > pros_duration:
        pad_vec = pros_rep[:, -1, :]
        pros_rep = torch.cat([pros_rep, pad_vec.unsqueeze(1).expand(1, mel_duration - pros_duration, pros_rep.size(2))], dim = 1)
    
    spk_emb = spk_enc_func(spk_enc, args.target_wav_list)
    spk_emb_tensor = torch.FloatTensor(spk_emb).unsqueeze(0).unsqueeze(0).to(args.device)

    # generate mel
    decoder_out = eval(dec_func)(dec, ling_rep, pros_rep, spk_emb_tensor)
    wav = eval(voc_func)(voc, decoder_out)
    os.makedirs(os.path.dirname(args.output_wav_path), exist_ok = True)
    sf.write(args.output_wav_path, wav.data.cpu().numpy(), 24000, "PCM_16")
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_ckpt', type = str)
    parser.add_argument('--model_config', type = str)
    parser.add_argument('--source_wav', type = str)
    parser.add_argument('--target_wav_list', type = str, nargs='+')
    parser.add_argument('--output_wav_path', type = str)
    parser.add_argument('--device', type = str, default = 'cpu')

    args = parser.parse_args()

    # load models
    modules = load_model(args)
    vc_fn(args, *modules)


 
