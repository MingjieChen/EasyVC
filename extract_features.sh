#!/bin/bash

# conda env
conda=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7
conda_env=torch_1.9
source $conda/bin/activate $conda_env

# stage
stage=4
stop_stage=4

# set up
dataset=vctk # vctk or libritts
linguistic_encoder=vqwav2vec 
speaker_encoder=utt_dvec
prosodic_encoder=ppgvc_f0
decoder=fastspeech2
vocoder=ppgvc_hifigan
. bin/parse_options.sh || exit 1;

# decide feature_type based on choices of vocoder and decoder
if [ "$vocoder" == "ppgvc_hifigan" ]; then
    feature_type=ppgvc_mel # mel, vits_spec, ppgvc_mel
elif [ "$decoder" == "vits"]; then
    feature_type=vits_spec
else
    feature_type=mel    
fi        

if [ "$dataset" == "vctk" ]; then
    train_split=train_nodev_all
    dev_split=dev_all
    eval_split=eval_all
    splits="train_nodev_all dev_all eval_all"
elif [ "$dataset" == "libritts" ]; then
    train_split=train_nodev_clean
    dev_split=dev_clean
    eval_split=eval_clean
    splits="train_nodev_clean dev_clean eval_clean"    
else
    exit 1;    
fi    




# step 1: spectrogram extraction
if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    echo "extract $feature_type for $dataset"
    ./bin/feature_extraction.sh $dataset $feature_type "$splits"
    echo "done!"
    if [ "$feature_type" == "mel" ]; then
        # normalize as parallel_wavegan
        echo "compute_statistics $feature_type for $dataset"
        stats_path=dump/$dataset/$train_split/$feature_type/${train_split}.npy
        ./bin/compute_statistics.sh $dataset $train_split $feature_type
        echo "done!"
        echo "normalize $feature_type for $dataset"
        ./bin/normalize.sh $dataset "$splits" $feature_type $stats_path   
        echo "done!"
    fi    
fi   

# step 2: linguistic representation extraction
if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    if [ "$linguistic_encoder" == "vqwav2vec" ] && [ ! -e ling_encoder/vqwav2vec/vq-wav2vec_kmeans.pt ]; then 
        echo "downloading vqwav2vec model checkpoint"
        mkdir -p ling_encoder/vqwav2vec 
        cd ling_encoder/vqwav2vec
        wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/vq-wav2vec_kmeans.pt
        cd ../..
        echo "done!"
    fi   
    echo "start extracting $linguistic_encoder representations" 
    ./bin/${linguistic_encoder}_feature_extraction.sh "$splits" $dataset
    echo "done!"
fi

# step 3: prosodic representation extraction
if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    echo "extract $prosodic_encoder for $dataset"
    ./bin/feature_extraction.sh $dataset $prosodic_encoder $splits
    echo "done!"
    if [ "$prosodic_encoder" == "fastspeech2_pitch_energy"  ]; then
        # normalize pitch & energy 
        echo "compute_statistics $prosodic_encoder for $dataset"
        stats_path=dump/$dataset/$train_split/$prosodic_encoder/${train_split}.npy
        ./bin/compute_statistics.sh $dataset $train_split $prosodic_encoder  
        echo "done!"
        echo "normalize $prosodic_encoder for $dataset"
        ./bin/normalize.sh $dataset "$splits" $prosodic_encoder $stats_path
        echo "done!"
    fi    
fi    

# step 4: speaker representation extraction
if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ]; then
    ./bin/d_vector_extract_utterance_embedding.sh "$splits"   $dataset
fi    
