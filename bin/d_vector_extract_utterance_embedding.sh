#!/bin/bash


conda=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7
conda_env=torch_1.7
source $conda/bin/activate $conda_env
root=$PWD
cd model/transformer_adversarial

#utters=$( ls /share/mini1/res/t/vc/studio/timap-en/vcc2020/baseline/vcc2020/groundtruth/*/E3*.wav)
#utters=$(ls $root/exp/transformer_adversarial/0310_ta_0/zs_vcc_converted_wavs_100/*/*.wav)
#utters=$(ls $root/exp/transformer_adversarial/0310_ta_0/zs_vcc_mean_wavs_60/*/*.wav)
#for utt in $utters ; do
#    spk=$( basename $( dirname $utt )| cut -d'_' -f2 )
#    base=$( basename $utt | sed "s/.wav//")
#    echo "$utt $spk $base"
#    python extract_utter_embed.py \
#       $utt \
#       $root/exp/transformer_adversarial/0310_ta_0/zs_vcc_converted_wavs_100/vcc_mean_spkembs/${base}.npy \
#       speaker_encoder/ckpt/pretrained_bak_5805000.pt  \
#       #$root/dump/vcc2020-spks/${spk}_${base}.npy \
#done       


#scp=/share/mini1/res/t/vc/studio/tiresyn-en/libritts/ParallelWaveGAN/egs/libritts/voc1/data/train_nodev_clean/wav.scp
#data_root=/share/mini1/res/t/vc/studio/tiresyn-en/libritts/ParallelWaveGAN/egs/libritts/voc1
#wavs=$(cat $scp | awk '{print $2}')

#for _wav in $wavs; do
#    fid=$(basename $_wav | cut -d '.' -f 1)
#    spk=$(echo $fid | cut -d '_' -f 1 )
#    echo "$data_root/$_wav $root/dump/utt_d_vec/$spk/${fid}.npy" >> speaker_embedding_meta.txt
    
#done

python extract_utter_embed.py speaker_embedding_meta.txt  speaker_encoder/ckpt/pretrained_bak_5805000.pt 24000


