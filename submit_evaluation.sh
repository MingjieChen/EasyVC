#!/bin/bash

train_dataset=libritts
eval_dataset=vctk
eval_split=eval_all
splits="train_nodev_all dev_all eval_all"
# eval step
step=all # utmos|asr|asv

# model setup
ling_enc=vqwav2vec
spk_enc=uttdvec
pros_enc=ppgvcf0
dec=tacoar
vocoder=ppgvchifigan
n_asr_jobs=60
# exp setup
exp_name=libritts_train_0
    
root=$PWD
# eval setup
task=oneshot_vc
epochs=92
eval_list=data/$eval_dataset/$eval_split/eval_list_m2m_vc_small_oneshot.json
. ./bin/parse_options.sh || exit 1;


exp_dir=exp/${train_dataset}_${ling_enc}_${spk_enc}_${pros_enc}_${dec}_${vocoder}/${exp_name}
if [ ! -e $exp_dir ] ; then
    echo "$exp_dir does not exist"
    exit 1;
fi
eval_wav_dir=$exp_dir/inference_${eval_dataset}/$task/$epochs



[ ! -e $exp_dir/evaluation_$eval_dataset ] && mkdir -p $exp_dir/evaluation_$eval_dataset

if [[ "$step" == "utmos" ]] || [[ "$step" == "all" ]]; then

    # utmos
    utmos_job=$exp_dir/scripts/utmos_${task}_${epochs}_${eval_dataset}.sh
    utmos_log=$exp_dir/logs/utmos_${task}_${epochs}_${eval_dataset}.log
    touch $utmos_job
    chmod +x $utmos_job

    cat <<EOF > $utmos_job
#!/bin/bash
wav_dir=$root/$eval_wav_dir
out_csv=$root/$exp_dir/evaluation_${eval_dataset}/utmos_${task}_${epochs}.csv
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
    # generate test_csv for speechbrain_asr.py from eval_wav_dir
    python evaluation/test_csv_speechbrain_asr.py \
            --eval_list $eval_list \
            --eval_wav_dir $root/$eval_wav_dir \
            --test_csv_path $root/$exp_dir/evaluation_${eval_dataset}/speechbrain_asr_test_csv_${task}_${epochs}.csv \
            --n_splits $n_asr_jobs
    asr_job=$exp_dir/scripts/asr_${task}_${epochs}_${eval_dataset}.sh
    asr_log=$exp_dir/logs/asr_${task}_${epochs}_${eval_dataset}.log
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

python evaluation/new_speechbrain_asr.py  evaluation/transformer_asr.yaml  \
        --test_csv=[$root/$exp_dir/evaluation_${eval_dataset}/speechbrain_asr_test_csv_${task}_${epochs}.\${SGE_TASK_ID}.csv]  \
        --wer_file=$root/$exp_dir/evaluation_${eval_dataset}/wer_${task}_${epochs}.\${SGE_TASK_ID}.txt \
        --output_folder=$root/$exp_dir/evaluation_${eval_dataset}/asr_out_${task}_${epochs}/output.\${SGE_TASK_ID} \
        --device=cpu
EOF
                            
    submitjob -n $n_asr_jobs -m 20000 -M2 $asr_log $asr_job
    echo "asr job submited, see ${asr_log}"
        
fi       
 
if [[ "$step" == "asr_score" ]]; then
    output_dirs=""
    for n in $(seq $n_asr_jobs); do
        output_dirs+=" $root/$exp_dir/evaluation_${eval_dataset}/asr_out_${task}_${epochs}/output.${n}"
    done
    job=$root/$exp_dir/scripts/asr_score_${eval_dataset}.sh
    log=$root/$exp_dir/logs/asr_score_${eval_dataset}.log
    touch $job
    chmod +x $job
    cat << EOF > $job
#!/bin/bash
conda=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7
conda_env=speechbrain
source \$conda/bin/activate \$conda_env           
python3 evaluation/speechbrain_asr_wer.py \
        --decoding_output_dirs $output_dirs \
        --decoding_output_filename token.txt \
        --wer_output_path $root/$exp_dir/evaluation_${eval_dataset}/asr_out_${task}_${epochs}/wer.all.txt
EOF
    submitjob -m 10000 $log $job        
    echo "asr score job submited, see logs in $log"
fi     


if [[ "$step" == "asv" ]] || [[ "$step" == "all" ]]; then
    
    eval_list_basename=$( basename $eval_list | sed -e "s/\.[^\.]*$//g")
    positive_pairs=evaluation/${eval_list_basename}_${eval_dataset}_asv_positive.txt
    negative_pairs=evaluation/${eval_list_basename}_${eval_dataset}_asv_negative.txt
    if [ ! -e $positive_pairs ] || [ ! -e $negative_pairs ]; then
        python3 evaluation/test_csv_speechbrain_asv_${eval_dataset}.py \
                --eval_list $eval_list \
                --positive_output_veri_file $positive_pairs \
                --negative_output_veri_file $negative_pairs \
                --data_dir data/$eval_dataset \
                --splits $splits  || exit 1;
    fi            
    asv_job=$exp_dir/scripts/asv_${task}_${epochs}_${eval_dataset}.sh
    asv_log=$exp_dir/logs/asv_${task}_${epochs}_${eval_dataset}.log
    [ -e $exp_dir/evaluation_${eval_dataset}/asv_out_${task}_${epochs} ] && rm -rf $exp_dir/evaluation_${eval_dataset}/asv_out_${task}_${epochs}
    mkdir -p $exp_dir/evaluation_${eval_dataset}/asv_out_${task}_${epochs}
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
           $negative_pairs \
           cuda \
           $exp_dir/evaluation_${eval_dataset}/asv_out_${task}_${epochs}
                

EOF
    submitjob  -g1 -m 20000 -M2 -o -l gputype="GeForceGTXTITANX|GeForceGTX1080Ti" -eo $asv_log $asv_job
    echo "asv job submited, see ${asv_log}"

fi    
