# Feature extraction

This part aims to extract features from speech data, including
 - mel-spectrogram 
  - ppgvc_mel
  - mel
 - linguistic representations
  - vq-wav2vec
  - conformer_ppg
  - hubert_soft and hubert_discrete	
 - utterance level speaker representations
  - d-vector
 - prosodic representations
  - ppgvc_logf0
  - fastspeech2 pitch + energy



## Mel-Spectrograms

We provides two types of implementations of mel-spectrograms:
	- ppgvc_mel: logmel_spectrograms from [ppg-vc](https://github.com/liusongxiang/ppg-vc)
	- mel: logmel_spectrogram from [parallel_wave_gan](https://github.com/kan-bayashi/ParallelWaveGAN)


### ppgvc_mel
It works compatible with the ppgvc_hifigan vocoder (from [ppg-vc](https://github.com/liusongxiang/ppg-vc)). It uses a min max normalization and is trained on VCTK dataset.
```
./bin/feature_extraction_ppgvc_mel_f0.sh
```

### mel

## Linguistic representation extraction

### vq-wav2vec
```

```