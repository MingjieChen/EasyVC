from .ppgvc_f0.ppgvc_lf0 import get_converted_lf0uv, compute_mean_std, compute_f0, f02lf0
from .fastspeech2_pitch_energy.pitch_energy import extract_pitch_energy
import torch
import librosa
import yaml
from sklearn.preprocessing import StandardScaler

def infer_norm_fastspeech2_pitch_energy(source_wav, target_wav = None, config_path = 'configs/preprocess_fastspeech2_pitch_energy.yaml', stats = 'dump/vctk/train_nodev_all/fastspeech2_pitch_energy/train_nodev_all.npy'):
    # load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
        f.close()
    # extract pitch energy    
    src_wav, _ = librosa.load(source_wav, sampling_rate = config['sampling_rate'])
    pitch_energy = extract_pitch_energy(src_wav, config)
    pitch = pitch_energy[0, :]
    energy = pitch_energy[1, :]
    # load pitch energy mean std
    scaler_pitch = StandardScaler()
    scaler_energy = StandardScaler()
    pitch_energy_stats = np.load(stats_path)
    scaler_pitch.mean_ = pitch_energy_stats[0]
    scaler_pitch.scale_ = pitch_energy_stats[1]
    scaler_energy.mean_ = pitch_energy_stats[2]
    scaler_energy.scale_ = pitch_energy_stats[3]
    scaler_pitch.n_features_in_ = scaler_pitch.mean_.shape[0]
    scaler_energy.n_features_in_ = scaler_energy.mean_.shape[0]

    # normalize pitch energy
    norm_pitch = scaler_pitch.transform(pitch.reshape(-1,1))
    norm_energy = scaler_energy.transform(energy.reshape(-1,1))
    output = np.array([norm_pitch.reshape(-1), norm_energy.reshape(-1)]).T
    output_tensor = torch.FloatTensor([output])
    return output_tensor

    
def infer_ppgvc_f0(source_wav, target_wav, config_path = 'configs/preprocess_ppgvc_mel.yaml', stats = None):
    with open(config_path) as f:
        config = yaml.safe_load(f)
        f.close()
    ref_wav, _ = librosa.load(target_wav, sr=config['sampling_rate'])
    ref_lf0_mean, ref_lf0_std = compute_mean_std(f02lf0(compute_f0(ref_wav)))
    src_wav, _ = librosa.load(source_wav, sr=sampling_rate)
    lf0_uv = get_converted_lf0uv(src_wav, ref_lf0_mean, ref_lf0_std, convert=True)
    lf0_uv = torch.FloatTensor([lf0_uv])
    return lf0_uv
