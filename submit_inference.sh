#!/bin/bash

dataset=vctk
split=eval_all
# model setup
ling_enc=vqwav2vec
spk_enc=uttdvec
pros_enc=ppgvcf0
dec=gradtts
vocoder=ppgvchifigan

# exp setup
exp_name=vctk_first_train
exp_dir=exp/${dataset}_${ling_enc}_${spk_enc}_${pros_enc}_${dec}_${vocoder}/${exp_name}


# eval setup
#task=oneshot_vc
task=m2m_vc
epochs=$( ls -t $exp_dir/ckpt | head -n 1 | sed 's/[^0-9]*//g')
eval_list=eval_list_m2m_vc_small.json
eval_list_path=data/$dataset/$split/$eval_list
# sge submitjob setup
n_parallel_jobs=50
device=cpu
job=$exp_dir/scripts/inference_${task}_${epochs}.sh
log=$exp_dir/logs/inference_${task}_${epochs}.log
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

EOF

#submit to sge
submitjob -m 20000 -n $n_parallel_jobs   $log $job
echo "job submitted, see log in ${log}"
