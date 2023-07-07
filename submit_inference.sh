#!/bin/bash

train_dataset=libritts
infer_dataset=vctk
split=eval_all
# model setup
ling_enc=vqwav2vec
spk_enc=uttdvec
pros_enc=ppgvcf0
dec=fs2
vocoder=ppgvchifigan

# exp setup
exp_name=libritts_train_0
if [ ! -e $exp_dir ]; then
    echo "$exp_dir does not exist"
    exit 1;
fi    

# eval setup
task=oneshot_vc
#task=m2m_vc

#epochs=210
eval_list=eval_list_m2m_vc_small_oneshot.json
eval_list_path=data/$infer_dataset/$split/$eval_list
# sge submitjob setup
n_parallel_jobs=100
device=cpu

. bin/parse_options.sh || exit 1;

exp_dir=exp/${train_dataset}_${ling_enc}_${spk_enc}_${pros_enc}_${dec}_${vocoder}/${exp_name}
epochs=$( ls -t $exp_dir/ckpt | head -n 1 | sed 's/[^0-9]*//g')


job=$exp_dir/scripts/inference_${task}_${epochs}_${infer_dataset}.sh
log=$exp_dir/logs/inference_${task}_${epochs}_${infer_dataset}.log
touch $job
chmod +x $job

# create bash file
cat <<EOF > $job

#!/bin/bash
conda=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7
conda_env=torch_1.9
source \$conda/bin/activate \$conda_env
echo "sge_task_id \$SGE_TASK_ID"
python inference.py \
        --exp_dir $exp_dir \
        --eval_list $eval_list_path \
        --epochs ${epochs} \
        --task ${task} \
        --device ${device} \
        --sge_task_id \$SGE_TASK_ID \
        --sge_n_tasks ${n_parallel_jobs} \
        --infer_dataset ${infer_dataset}

EOF


#submit to sge
submitjob -m 40000 -n $n_parallel_jobs -o -l hostname="!node20&!node21&!node23&!node24&!node26&!node27&!node28&!node29" -eo   $log $job
echo "job submitted, see log in ${log}"
