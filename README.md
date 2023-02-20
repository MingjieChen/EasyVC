# enc_dec_voice_conversion (**EDVC**)

**Work in progress.**

A voice conversion framework for different types of encoders, decoders and vocoders. 

The encoder-decoder framework is demonstrated in the following figure. ![figure](enc_dec_voice_conversion.drawio.png)

More specifically, three encoders are used to extract representations from speech, including a linguistic encoder, a prosodic encoder and a speaker encoder.
Then a decoder is used to reconstruct speech mel-spectrograms. 
Finally, a vocoder converts mel-spectrograms to waveforms. 
Note that this repo also supports decoders that directly reconstruct waveforms (e.g. VITS), in these case, vocoders are not needed. 


This repo covers all the steps of a voice conversion pipeline from dataset downloading to evaluation.

I am currently working on my own to maintain this repo. I am planning to integrate more encoders and decoders.

Pleas be aware that this repo is currently very unstable and under very fast developement.


# Conda env

create a conda env
```
conda create --name torch_1.9 --file requirements.txt
```



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
## Step2: Generate metadata.csv

```
./bin/preprocess_vctk.sh
```
Or
```
./bin/preprocess_libritts.sh
```

## Step3: Extract features

A ESPNET style bash script has been provided for extracting features, including spectrograms, linguistic, speaker, and prosodic representations.
e.g.
```
./extract_features.sh --stage 1 \
                      --stop_stage 4 \
                      --dataset vctk \
                      --mel_type ppgvc_mel \
                      --linguistic_encoder vqwav2vec \
                      --speaker_encoder utt_dvec \
                      --prosodic_encoder ppgvc_f0
```

## Step4: Training

more to come
