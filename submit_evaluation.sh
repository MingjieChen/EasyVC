#!/bin/bash

dataset=vctk
eval_split=eval_all
# eval step
step=asv # utmos|asr|asv

# model setup
ling_enc=conformerppg
spk_enc=uttdvec
pros_enc=ppgvcf0
dec=vits
vocoder=none

# exp setup
exp_name=vctk_first_train
exp_dir=exp/${dataset}_${ling_enc}_${spk_enc}_${pros_enc}_${dec}_${vocoder}/${exp_name}
if [ ! -e $exp_dir ] ; then
    echo "$exp_dir does not exist"
    exit 1;
fi
    
root=$PWD
# eval setup
task=m2m_vc
epochs=146
eval_list=data/$dataset/eval_all/eval_list_m2m_vc_small_oneshot.json
eval_wav_dir=$exp_dir/inference/$task/$epochs

[ ! -e $exp_dir/evaluation ] && mkdir -p $exp_dir/evaluation

if [[ "$step" == "utmos" ]] || [[ "$step" == "all" ]]; then

    # utmos
    utmos_job=$exp_dir/scripts/utmos_${task}_${epochs}.sh
    utmos_log=$exp_dir/logs/utmos_${task}_${epochs}.log
    touch $utmos_job
    chmod +x $utmos_job

    cat <<EOF > $utmos_job
#!/bin/bash
wav_dir=$root/$eval_wav_dir
out_csv=$root/$exp_dir/evaluation/utmos_${task}_${epochs}.csv
conda=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7
conda_env=voicemos
source \$conda/bin/activate \$conda_env

cd evaluation/UTMOS-demo/
echo "eval_wav_dir: \${wav_dir} out_csv \${out_csv}"
python predict.py --mode predict_dir --inp_dir \$wav_dir --bs 1 --out_path \$out_csv

EOF

    submitjob -m 20000 -M2 $utmos_log $utmos_job
    echo "utmos job submited, see ${utmos_log}"
fi

# speechbrain asr

if [[ "$step" == "asr" ]] || [[ "$step" == "all" ]]; then
    if [ ! -e $root/$exp_dir/evaluation/speechbrain_asr_test_csv_${task}_${epochs}.csv ]; then
        # generate test_csv for speechbrain_asr.py from eval_wav_dir
        python evaluation/test_csv_speechbrain_asr.py \
            --eval_list $eval_list \
            --eval_wav_dir $root/$eval_wav_dir \
            --test_csv_path $root/$exp_dir/evaluation/speechbrain_asr_test_csv_${task}_${epochs}.csv
    fi        
    asr_job=$exp_dir/scripts/asr_${task}_${epochs}.sh
    asr_log=$exp_dir/logs/asr_${task}_${epochs}.log
    touch $asr_job
    chmod +x $asr_job
    cat <<EOF > $asr_job
#!/bin/bash
conda=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7
conda_env=speechbrain
source \$conda/bin/activate \$conda_env

export PATH=/share/mini1/sw/std/cuda/cuda11.1/bin:\$PATH
export CUDA_HOME=/share/mini1/sw/std/cuda/cuda11.1/
export LD_LIBRARY_PATH=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7/envs/StyleSpeech/lib:/share/mini1/sw/std/cuda/cuda11.1/lib64:\$LD_LIBRARY_PATH

python evaluation/speechbrain_asr.py  evaluation/speechbrain_asr.yaml  \
        --test_csv=[$root/$exp_dir/evaluation/speechbrain_asr_test_csv_${task}_${epochs}.csv]  \
        --wer_file=$root/$exp_dir/evaluation/wer_${task}_${epochs}.txt \
        --output_folder=$root/$exp_dir/evaluation/asr_out_${task}_${epochs} \
        --device=cpu
EOF
                            
    submitjob -m 20000 -M2 $asr_log $asr_job
    echo "asr job submited, see ${asr_log}"
        
fi        


if [[ "$step" == "asv" ]] || [[ "$step" == "all" ]]; then
    eval_list_basename=$( basename $eval_list | sed -e "s/\.[^\.]*$//g")
    positive_pairs=evaluation/${eval_list_basename}_asv_positive.txt
    negative_pairs=evaluation/${eval_list_basename}_asv_negative.txt
    if [ ! -e $positive_pairs ] || [ ! -e $negative_pairs ]; then
        python3 evaluation/test_csv_speechbrain_asv.py \
                --eval_list $eval_list \
                --positive_output_veri_file $positive_pairs \
                --negative_output_veri_file $negative_pairs \
                --data_dir data/$dataset \
                --splits train_nodev_all dev_all eval_all
    fi            
    asv_job=$exp_dir/scripts/asv_${task}_${epochs}.sh
    asv_log=$exp_dir/logs/asv_${task}_${epochs}.log
    touch $asv_job
    chmod +x $asv_job
    cat <<EOF > $asv_job
conda=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7
conda_env=speechbrain
source \$conda/bin/activate \$conda_env
export PATH=/share/mini1/sw/std/cuda/cuda11.1/bin:\$PATH
export CUDA_HOME=/share/mini1/sw/std/cuda/cuda11.1/
export LD_LIBRARY_PATH=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7/envs/StyleSpeech/lib:/share/mini1/sw/std/cuda/cuda11.1/lib64:\$LD_LIBRARY_PATH
python3 evaluation/speechbrain_asv.py \
           $eval_wav_dir \
           $positive_pairs \
           $negative_pairs 
                

EOF
    submitjob  -g1 -m 20000 -M2 $asv_log $asv_job
    echo "asv job submited, see ${asv_log}"

fi    
