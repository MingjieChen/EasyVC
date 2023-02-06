import argparse
import json
import os
import sys
import glob
import random
import csv
import re

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    # task
    parser.add_argument('--task', type = str, choices = ['vc', 'resyn'])
    parser.add_argument('--split', type = str)
    parser.add_argument('--spk_enc', type = str)
    # path
    parser.add_argument('--speakers_path', type = str)
    parser.add_argument('--eval_metadata_path', type = str)
    parser.add_argument('--eval_list_out_path', type = str)
    # task setup
    
    n_samples_per_trg_speaker = parser.add_argument('--n_samples_per_trg_speaker', type = int)
    n_eval_speakers = parser.add_argument('--n_eval_speakers', type = int)
    n_samples_per_src_speaker = parser.add_argument('--n_samples_per_src_speaker', type = int)
    
    
    args = parser.parse_args()
    # load in all speakers in eval set
    with open(args.speakers_path) as f:
        speakers = f.readlines()
        speakers = [spk.strip() for spk in speakers]
        f.close()
    
    

    # load eval metadata csv
    metadata = []
    with open(args.eval_metadata_path) as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            metadata.append(row)
        f.close()
    
    # construct spk2wavs
    spk2wavs = {}
    for meta in metadata:
        ID = meta['ID']
        spk =  meta['spk']         
        if spk not in spk2wavs:
            spk2wavs[spk] = []
        
        spk2wavs[spk].append(meta)     
    
    selected_speakers = []
    for spk in speakers:
        if spk not in spk2wavs:
            print(f'spk {spk} not in metadata')
        else:    
            n_spk_samples = len(spk2wavs[spk])
            print(f'spk {spk} obtained {n_spk_samples} samples')    
            if n_spk_samples < args.n_samples_per_src_speaker or n_spk_samples < args.n_samples_per_trg_speaker:
                continue
            selected_speakers.append(spk)    
    
    print(f'after filtering, there remains {len(selected_speakers)} eval speakers')        
                
    if len(selected_speakers) > args.n_eval_speakers:
        # random sample speakers
        selected_speakers = random.sample(selected_speakers, k = args.n_eval_speakers)
    # start sample from eval set
    # loop over speakers
    eval_list = []
    selected_src_metas = {}
    selected_trg_metas = {}

    # sample spk metas
    for spk in selected_speakers:
        print(spk)
        selected_src_metas[spk] = []
        selected_trg_metas[spk] = []
        _spk_metas = spk2wavs[spk]
        _src_spk_metas_idxs = random.sample(range(0,len(_spk_metas)), k = int(args.n_samples_per_src_speaker))
        _trg_spk_metas_idxs = random.sample(range(0,len(_spk_metas)), k = int(args.n_samples_per_trg_speaker))
        _selected_src_spk_metas = [ _spk_metas[_i] for _i in _src_spk_metas_idxs]
        _selected_trg_spk_metas = [ _spk_metas[_i] for _i in _trg_spk_metas_idxs]
        selected_src_metas[spk].extend(_selected_src_spk_metas)
        selected_trg_metas[spk].extend(_selected_trg_spk_metas)
        print(f'spk {spk}| src: {len(_selected_src_spk_metas)}, trg: {len(_selected_trg_spk_metas)}')
   
        
    # produce eval list
    if args.task == 'vc':
        for src_spk in selected_speakers:
            src_metas = selected_src_metas[src_spk]
            for trg_spk in selected_speakers:
                if src_spk != trg_spk:
                    trg_metas = selected_trg_metas[trg_spk]
                    trg_wavs = [_trg_meta['wav_path'] for _trg_meta in trg_metas]
                    for _meta in src_metas:
                        
                        ID = _meta['ID']
                        src_wav = meta['wav_path']
                        duration = meta['duration']
                        text = meta['wrd']
                        src_spk = meta['spk']

                        element = {
                                    'ID': ID + '_' + trg_spk,
                                    'duration': duration,
                                    'text': text,
                                    'src_spk': src_spk,
                                    'trg_spk': trg_spk,
                                    'src_wav': src_wav,
                                    'trg_wav': []
                                     
                                  }  
                        element['trg_wav'].extend(trg_wavs)
                        eval_list.append(element)            
    elif args.task == 'resyn':
        for src_spk in selected_speakers:
            trg_spk = src_spk
            src_metas = selected_src_metas[src_spk]
            trg_metas = selected_trg_metas[trg_spk]
            for _meta in src_metas:                        
                ID = _meta['ID']
                src_wav = meta['wav_path']
                duration = meta['duration']
                text = meta['wrd']
                src_spk = meta['spk']

                element = {
                            'ID': ID + '_' + trg_spk,
                            'duration': f'{duration:.2f}',
                            'text': text,
                            'src_spk': src_spk,
                            'trg_spk': trg_spk,
                            'src_wav': src_wav,
                            'trg_wav': []
                             
                          }  
                element['trg_wav'].extend(trg_wavs)
                eval_list.append(element)            
    else:
        raise Exception                
                                         
        
    with open(args.eval_list_out_path, mode="w") as f:

        json.dump(eval_list, f, indent = 4)
        f.close()
    
            
        
                
