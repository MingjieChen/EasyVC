import os
import random
import json
from scipy.interpolate import interp1d
import librosa
import numpy as np
import pyworld as pw

def process_norm_fastspeech2_pitch_energy(pitch_energy):
    return pitch_energy.T
def extract_energy(
    audio,
    sampling_rate,
    fft_size=1024,
    hop_size=256,
    win_length=None,
    window="hann",
    num_mels=80,
    fmin=None,
    fmax=None,
    eps=1e-10,
    log_base=10.0,
):
    # get amplitude spectrogram
    x_stft = librosa.stft(
        audio,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=win_length,
        window=window,
        pad_mode="reflect",
    )
    spc = np.abs(x_stft).T  # (#frames, #bins)
    energy = np.linalg.norm(spc, axis = 1, ord = 2)
    return energy

def extract_pitch_energy(audio, config):
    pitch, t = pw.dio(
            audio.astype(np.float64),
            config['sampling_rate'],
            frame_period=config['hop_size'] / config['sampling_rate'] * 1000,
            )
    pitch = pw.stonemask(audio.astype(np.float64), pitch, t, config['sampling_rate'])
    #nonzero_ids = np.where(pitch != 0)[0]
    #interp_fn = interp1d(
    #        nonzero_ids,
    #        pitch[nonzero_ids],
    #        fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
    #        bounds_error=False,
    #        )
    #pitch = interp_fn(np.arange(0, len(pitch)))
    energy = extract_energy(
        audio,
        sampling_rate=config['sampling_rate'],
        hop_size=config['hop_size'],
        fft_size=config["fft_size"],
        win_length=config["win_length"],
        window=config["window"],
        num_mels=config["num_mels"],
        fmin=config["fmin"],
        fmax=config["fmax"]
    )
    pitch_energy = np.array([pitch, energy])
    return pitch_energy

    
