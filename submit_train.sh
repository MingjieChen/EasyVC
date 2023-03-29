#!/bin/bash

#conda
conda=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7
conda_env=torch_1.9

#choose config
dataset=vctk
ling=vqwav2vec
#ling=conformerppg
#ling=contentvec100
#ling=whisperppgsmall

spk=uttdvec
#spk=uttecapatdnn

pros=ppgvcf0
#pros=fs2pitchenergy

#dec=fs2
#dec=vits
#dec=gradtts
dec=diffwave
#dec=tacoar
#dec=tacomol

#vocoder=ppgvchifigan
vocoder=none
#vocoder=bigvgan

exp_name=vctk_train_2
#exp_name=vctk_no16fp_split


config=configs/${dataset}_${ling}_${spk}_${pros}_${dec}_${vocoder}.yaml
if [ ! -e $config ] ; then
    echo "can't find config file $config" 
    exit 1;
fi    
exp_dir=exp
model_name=${dataset}_${ling}_${spk}_${pros}_${dec}_${vocoder}
exp=$exp_dir/$model_name/$exp_name
njobs=24
ngpus=1
slots=4
#gputypes="GeForceRTX3060|GeForceRTX3090"
gputypes="GeForceRTX3090"
#gputypes="GeForceGTXTITANX|GeForceGTX1080Ti|GeForceRTX3060"
#gputypes="GeForceGTX1080Ti"

# create exp dir
[ ! -e $exp ] && mkdir -p $exp
[ ! -e $exp/scripts ] && mkdir -p $exp/scripts
[ ! -e $exp/logs ] && mkdir -p $exp/logs 

job_dir=$exp/scripts
log_dir=$exp/logs
exp_config=$exp/$(basename $config)
[ ! -e $exp_config ] && cp $config $exp_config

#submit first job
#jid=$(submitjob -m 10000 -g${ngpus} -M${slots} -o -l gputype=$gputypes  -eo  $log_dir/train.log  ./bin/train.sh | grep -E [0-9]+)
jid=""
jobs_to_kill="qdel"
# create following jobs
for ((n=0;n<${njobs};n++)); do
    job=$job_dir/train${n}.sh
    cat <<EOF > $job
#!/bin/bash
    
source $conda/bin/activate $conda_env
export CUDA_HOME=/share/mini1/sw/std/cuda/cuda11.3/x86_64/
export LD_LIBRARY_PATH="\${CUDA_HOME}/lib64:\${LD_LIBRARY_PATH}"
exp=$exp
ckpt_dir=$exp/ckpt/
if [ ! -e \${ckpt_dir} ] ; then
    export NCCL_IB_DISABLE=1
    export NCCL_SOCKET_IFNAME=virbr0
    python train.py  \
        -c $exp_config \
        -e $exp_name \
        -l $exp_dir \
        -m $model_name 
else
    ckpt=\$(ls -t \$ckpt_dir/*.pth | head -n 1)     
    echo "resume from \$ckpt"
    export NCCL_IB_DISABLE=1
    export NCCL_SOCKET_IFNAME=virbr0
    python train.py  \
        -p \$ckpt \
        -c $exp_config \
        -e $exp_name \
        -l $exp_dir \
        -m $model_name 
fi
EOF

    chmod +x $job
    log=$log_dir/train.log
    if [[ "$jid" == "" ]] ; then
        jid=$(submitjob -m 10000  -g${ngpus} -M${slots} -o -l gputype=$gputypes  -eo  $log  $job | grep -E [0-9]+)
    else    
        jid=$(submitjob -m 10000 -w $jid  -g${ngpus} -M${slots} -o -l gputype=$gputypes  -eo  $log  $job | grep -E [0-9]+)
    fi
    jobs_to_kill+=" $jid"    
    echo "submit $jid job $job log $log"
done    
[ -e $job_dir/kill_all.sh ] && rm $job_dir/kill_all.sh
touch $job_dir/kill_all.sh; echo "$jobs_to_kill" >> $job_dir/kill_all.sh; chmod +x $job_dir/kill_all.sh

