#!/bin/bash


# model setup
ling_enc=vqw2v
spk_enc=uttdvec
pros_enc=none
dec=tacoar
vocoder=libritts_hifigan

# exp setup
exp_name=first_train
exp_dir=exp/${ling_enc}_${spk_enc}_${pros_enc}_${dec}/${exp_name}
root=$PWD
# eval setup
task=oneshot_vc
epochs=72
eval_list=data/libritts/eval_clean/eval_list_oneshot_vc_small.json
eval_wav_dir=$exp_dir/inference/$task/$epochs

[ ! -e $exp_dir/evaluation ] && mkdir -p $exp_dir/evaluation



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
echo "job submited, see ${utmos_log}"


