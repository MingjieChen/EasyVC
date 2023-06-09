import csv
import glob
import os
import sys
import argparse
import random
import json
import re


'''
    generate speaker verification files for vctk
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--eval_list', type = str)
    parser.add_argument('--positive_output_veri_file', type = str)
    parser.add_argument('--negative_output_veri_file', type = str)
    parser.add_argument('--data_dir', type = str)
    parser.add_argument('--splits', type = str, nargs = '+')
    

    args = parser.parse_args()
    

    # get real spk2wavs
    real_spk2wav = {}
    for split in args.splits:
        wav_scp_path = os.path.join(args.data_dir, split, 'wav.scp')
        with open(wav_scp_path) as f:
            for line in f:
                line = line.strip()
                ID = line.split()[0]
                spk = ID.split('_')[0]
                wav_path = line.split()[1]
                if spk not in real_spk2wav:
                    real_spk2wav[spk] = []
                real_spk2wav[spk].append(wav_path)    
            f.close()
        

    print(f'get real spk2wav done')

    
    positive_pairs = []
    negative_pairs = []

    n_samples = 10
    n_neg_spks = 10
    with open(args.eval_list) as f:
        eval_list = json.load(f)
        f.close()
    for eval_item in eval_list:

        ID = eval_item['ID']
        spk = ID.split('_')[0]
        
        n_pos_samples = n_samples if n_samples < len(real_spk2wav[spk]) else len(real_spk2wav[spk])
        positive_samples = random.sample(real_spk2wav[spk], n_pos_samples)
        for positive_sample in positive_samples:
            positive_pairs.append((ID, positive_sample))
            
        tmp_spks = list(real_spk2wav.keys())
        tmp_spks.remove(spk)
        neg_spks = random.sample(tmp_spks, n_neg_spks) 
        for neg_spk in neg_spks:
            negative_sample = random.sample(real_spk2wav[neg_spk], 1)[0]
            negative_pairs.append((ID, negative_sample))
    
    print(f'obtained {len(positive_pairs)} positive_pairs')        
    print(f'obtained {len(negative_pairs)} negative_pairs')        
    
    f = open(args.positive_output_veri_file, 'w')
    for pair in positive_pairs:
        ID1, ID2 = pair
        f.write(f'{ID1} {ID2}\n')
    f.close()
    
    f = open(args.negative_output_veri_file, 'w')
    for pair in negative_pairs:
        ID1, ID2 = pair
        f.write(f'{ID1} {ID2}\n')
    f.close()
        





        
