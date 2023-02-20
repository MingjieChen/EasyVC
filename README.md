# enc_dec_voice_conversion (**EDVC**)

**Work in progress.**

A voice conversion framework for different types of encoders and decoders. The encoder-decoder framework is demonstrated in the following figure ![figure](enc_dec_voice_conversion.drawio.png)

This repo covers all the pipelines from dataset downloading to evaluation.

# Working progress

- **Dataset**
 - [x] VCTK
 - [x] LibriTTS

- **Linguistic Encoder**
 - [x] conformer_ppg from [ppg-vc](https://github.com/liusongxiang/ppg-vc)
 - [x] vq-wav2vec from [fairseq](https://github.com/facebookresearch/fairseq)
 - [x] hubert_soft and hubert_discrete from [soft-vc](https://github.com/bshall/soft-vc)
 
 
- **Prosodic Encoder**
 - [x] log-f0 from [ppg-vc](https://github.com/liusongxiang/ppg-vc)
 - [x] pitch + energy from [fastspeech2](https://github.com/ming024/FastSpeech2)
 
 
- **Speaker Encoder**
 - [x] d-vector from [ppg-vc](https://github.com/liusongxiang/ppg-vc)
 
 
- **Decoder**
 - [x] fastspeech2 from [fastspeech2](https://github.com/ming024/FastSpeech2)
 - [x] taco_ar from [s3prl-vc](https://github.com/s3prl/s3prl/tree/main/s3prl/downstream/a2a-vc-vctk)
 - [x] taco_mol from [ppg-vc](https://github.com/liusongxiang/ppg-vc)
 - [x] vits from [vits](https://github.com/jaywalnut310/vits)
 
 
- **Vocoder**
 - [x] hifigan (vctk) from [ppg-vc](https://github.com/liusongxiang/ppg-vc)

- **Evaluation**
 - [ ] UTMOS22 mos prediction from [UTMOS22](https://github.com/sarulab-speech/UTMOS22)
 - [ ] ASR WER
 - [ ] ASV EER
 - [ ] MCD, F0-RMSE, F0-CORR
# How to run

## Step1: Dataset download 
This part of codes are mostly from [parallel_wavegan](https://github.com/kan-bayashi/ParallelWaveGAN)

```
./bin/download_vctk_dataset.sh
```

Or

```
./bin/download_libritts_dataset.sh
```
## Step2: Preprocessing (generate metadata.csv)

```
./bin/preprocess_vctk.sh
```
Or
```
./bin/preprocess_libritts.sh
```
