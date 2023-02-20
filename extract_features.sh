#!/bin/bash

# conda env
conda=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7
conda_env=torch_1.9
source $conda/bin/activate $conda_env

# stage
stage=
stop_stage=

# set up
dataset=vctk # vctk or libritts
mel_type=ppgvc_mel # mel, vits_spec, ppgvc_mel
linguistic_encoder=vqwav2vec 
speaker_encoder=utt_dvec
prosodic_encoder=ppgvc_f0
decoder=fastspeech2
vocoder=ppgvc_hifigan

if [ "$dataset" == "vctk" ]; then
    splits="train_nodev_all dev_all eval_all"
elif [ "$dataset" == "libritts" ]; then
    splits="train_nodev_clean dev_clean eval_clean"    
else
    exit 1;    
fi    


. bin/parse_options.sh || exit 1;


# step 1: spectrogram extraction
if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    ./bin/feature_extraction.sh $dataset $feature_type
fi   

# step 2: linguistic representation extraction
if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    if [ "$linguistic_encoder" == "vqwav2vec" ]; then 
        mkdir -p ling_encoder/vqwav2vec 
        cd ling_encoder/vqwav2vec
        wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/vq-wav2vec_kmeans.pt
        cd ../..
    fi    
    ./bin/${linguisti_encoder}_feature_extraction.sh $splits $dataset
fi

# step 3: prosodic representation extraction
if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    ./bin/feature_extraction.sh $dataset $prosodic_encoder
fi    

# step 4: speaker representation extraction
if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ]; then
    ./bin/d_vector_extract_utterance_embedding.sh $split   $dataset
fi    




   


 


