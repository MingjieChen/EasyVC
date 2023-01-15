#!/bin/bash

#conda
conda=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7
conda_env=torch_1.7

#exp setup
#ling=vqw2v
ling=conformerppg
spk=uttdvec
pros=none
#dec=fastspeech2
dec=tacoar

exp_name=first_train
config=configs/${ling}_${spk}_${pros}_${dec}.yaml
exp_dir=exp
model_name=${ling}_${spk}_${pros}_${dec}
exp=$exp_dir/$model_name/$exp_name
njobs=12
ngpus=2
slots=8
#gputypes="GeForceRTX3060|GeForceRTX3090"
gputypes="GeForceRTX3090"

# create exp dir
[ ! -e $exp ] && mkdir -p $exp
[ ! -e $exp/scripts ] && mkdir -p $exp/scripts
[ ! -e $exp/logs ] && mkdir -p $exp/logs 

job_dir=$exp/scripts
log_dir=$exp/logs
exp_config=$exp/$(basename $config)
cp $config $exp_config

#submit first job
#jid=$(submitjob -m 10000 -g${ngpus} -M${slots} -o -l gputype=$gputypes  -eo  $log_dir/train.log  ./bin/train.sh | grep -E [0-9]+)
jid=""
# create following jobs
for ((n=0;n<=${njobs};n++)); do
    job=$job_dir/train${n}.sh
    cat <<EOF > $job
#!/bin/bash
source $conda/bin/activate $conda_env
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
    log=$log_dir/train${n}.log
    if [[ "$jid" == "" ]] ; then
        jid=$(submitjob -m 10000  -g${ngpus} -M${slots} -o -l gputype=$gputypes  -eo  $log  $job | grep -E [0-9]+)
    else    
        jid=$(submitjob -m 10000 -w $jid  -g${ngpus} -M${slots} -o -l gputype=$gputypes  -eo  $log  $job | grep -E [0-9]+)
    fi    
    echo "submit $jid job $job log $log"
done    

