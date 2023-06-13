# **EasyVC**


> Current state-of-the-art voice conversion (VC) systems typically are developed based on an encoder-decoder framework. In this framework, encoders are used to extract linguistic, speaker or prosodic features from speech, then a decoder is to generate speech from speech features. Recently, there have been more and more advance models deployed as encoders or decoders for VC. Although obtaining good performance, the effects of these encoders and decoders have not been fully studied. On the other hand, VC technologies have been applied in different scenarios, which brings a lot of challenges for VC techiques. Hence, studies and understandings of encoders and decoders are becoming necessary and important. However, due to the complexity of VC systems, it is not always easy to compare and analyse these encoders and decoders. This paper introduces a toolkit, EasyVC, which is built upon the encoder-decoder framework. EasyVC supports a number of encoders and decoders within a unified framework, which makes it easy and convenient for VC training, inference, evaluation and deployment. EasyVC provides step-wise recipes covering from dataset downloading to objective evaluations and online demo presentation. Furthermore, EasyVC focuses on challenging VC scenarios such as one-shot, emotional, singing and real-time, which have not been fully studied at the moment. EasyVC could help researchers and developers to investigate modules of VC systems and also promote the development of VC techniques. 

***

[[demo-page](https://mingjiechen.github.io/easyvc/index.html)]


The encoder-decoder framework is demonstrated in the following figure. ![figure](enc_dec_voice_conversion.drawio.png)

More specifically, three encoders are used to extract representations from speech, including a linguistic encoder, a prosodic encoder and a speaker encoder.
Then a decoder is used to reconstruct speech mel-spectrograms. 
Finally, a vocoder converts mel-spectrograms to waveforms. 
Note that this repo also supports decoders that directly reconstruct waveforms (e.g. VITS), in these case, vocoders are not needed. 


This repo covers all the steps of a voice conversion pipeline from dataset downloading to evaluation.

Trained models will be available soon.



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
    - [x] hubert_soft from [soft-vc](https://github.com/bshall/soft-vc)
    - [x] contentvec_100 from [contentvec](https://github.com/auspicious3000/contentvec)
    - [x] contentvec_500 from [contentvec](https://github.com/auspicious3000/contentvec)
    - [x] whisper_ppg from [whisper_ppg](https://github.com/PlayVoice/whisper_ppg)
 
 
- **Prosodic Encoder**
    - [x] log-f0 from [ppg-vc](https://github.com/liusongxiang/ppg-vc)
    - [x] pitch + energy from [fastspeech2](https://github.com/ming024/FastSpeech2)
 
 
- **Speaker Encoder**
    - [x] d-vector from [ppg-vc](https://github.com/liusongxiang/ppg-vc)
    - [x] ECAPA-TDNN from [speechbrain](https://github.com/speechbrain/speechbrain/tree/develop/recipes/VoxCeleb)
 
 
- **Decoder**
    - [x] fastspeech2 from [fastspeech2](https://github.com/ming024/FastSpeech2)
    - [x] taco_ar from [s3prl-vc](https://github.com/s3prl/s3prl/tree/main/s3prl/downstream/a2a-vc-vctk)
    - [x] taco_mol from [ppg-vc](https://github.com/liusongxiang/ppg-vc)
    - [x] vits from [vits](https://github.com/jaywalnut310/vits)
    - [x] grad_tts from [Grad_TTS](https://github.com/huawei-noah/Speech-Backbones)
    - [x] diffwave from [DiffWave](https://github.com/lmnt-com/diffwave)
 
 
- **Vocoder**
    - [x] hifigan (vctk) from [ppg-vc](https://github.com/liusongxiang/ppg-vc)

- **Evaluation**
    - [x] UTMOS22 mos prediction from [UTMOS22](https://github.com/sarulab-speech/UTMOS22)
    - [x] ASR WER
    - [x] ASV EER
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
Before start extracting features, you need to decide the setups of your encoders, decoder and vocoder.

e.g.
```
./extract_features.sh --stage 1 \
                      --stop_stage 4 \
                      --dataset vctk \
                      --linguistic_encoder vqwav2vec \
                      --speaker_encoder utt_dvec \
                      --prosodic_encoder ppgvc_f0 \
                      --decoder fastspeech2 \
                      --vocoder ppgvc_hifigan
```
Options:
- dataset: 
    - vctk 
    - libritts
- speaker_encoder: 
    - utt_dvec
    - utt_ecapa_tdnn
- linguistic_encoder: 
    - vqwav2vec
    - conformer_ppg 
    - hubert_soft
    - contentvec_100
    - contentvec_500
    - whisper_ppg
- prosodic_encoder: 
    - ppgvc_f0 
    - fastspeech2_pitch_energy
- decoder:
    - fastspeech2
    - taco_ar
    - taco_mol
    - vits
- vocoder:
    - ppgvc_hifigan
    - vctk_hifigan
    - libritts_hifigan
    

## Step4: Training

To run training, you need to select a config file from `configs/`. 
The config files are named following the format `${dataset}_${linguistic_encoder}_${speaker_encoder}_${prosodic_encoder}_${decoder}_${vocoder}`
E.g.
```
./bin/train.sh configs/vctk_vqwav2vec_uttdvec_ppgvcf0_fs2_ppgvchifigan.yaml
```

## Step5: Generate Eval List & Inference

To generate a eval list, you need to run e.g.
```
./bin/generate_eval_list.sh --task vc \
                            --dataset vctk \
                            --split eval_all #eval set name\
                            --eval_list eval_list.json #eval_list file name \
                            --n_trg_spk_samples 10 #number of randomly selected samples of target speakers to get averaged speaker embedding \
                            --n_src_spk_samples 10 # number of randomly selected samples of source speakers to test \
                            --n_eval_spks # number of randomly selected speakers from eval set
```

Then run inference using the genertated eval_list, e.g.

```
python inference.py \
          --exp_dir exp/${dataset}_${ling_enc}_${spk_enc}_${pros_enc}_${dec}_${vocoder}/${exp_name} \
          --eval_list data/$dataset/$split/$eval_list \
          --epochs 200 \
          --task a2a_vc # a2a_vc, m2m_vc or oneshot_vc, will decide how many target speaker embeddings to be used. \
          --device cpu \

```

## Step6: Evaluation

```
./submit_evaluation.sh
```


# Authors

- Mingjie Chen, University of Sheffield
- Prof. Thomas Hain, University of Sheffield
