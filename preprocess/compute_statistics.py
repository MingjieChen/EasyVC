import argparse
import logging
import os
import csv

import numpy as np
import yaml

from sklearn.preprocessing import StandardScaler
from tqdm import tqdm



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dump_dir', type = str)
    parser.add_argument('--metadata', type = str)
    parser.add_argument('--split', type = str)
    parser.add_argument('--feature_type', type = str, default = 'mel')

    args = parser.parse_args()
    # create scaler
    if args.feature_type == 'fastspeech2_pitch_energy':
        scaler_pitch = StandardScaler()
        scaler_energy = StandardScaler()
    else:
        scaler = StandardScaler()    
    metadata = []
    # load metadata
    with open(args.metadata) as f:
        csv_reader = csv.DictReader(f)
        for meta in csv_reader:
            metadata.append(meta)
        f.close()
    for _meta in tqdm(metadata):        
        ID = _meta['ID']
        spk = _meta['spk']
        feature_path = os.path.join(args.dump_dir, args.split, args.feature_type, spk, ID+'.npy')
        assert os.path.exists(feature_path), f'{feature_path}'
        feature = np.load(feature_path)
        if args.feature_type == 'fastspeech2_pitch_energy':
            pitch = feature[0, :]
            energy = feature[1, :]
            scaler_pitch.partial_fit(pitch.reshape(-1, 1))
            scaler_energy.partial_fit(energy.reshape(-1, 1))
        else:    
            scaler.partial_fit(feature)


    out_path = os.path.join(args.dump_dir, args.split, args.feature_type, args.split + '.npy')    

    if args.feature_type == 'fastspeech2_pitch_energy':
        stats = np.stack([scaler_pitch.mean_, scaler_pitch.scale_, scaler_energy.mean_, scaler_energy.scale_], axis = 0)
    else:    
        stats = np.stack([scaler.mean_, scaler.scale_], axis=0)
    np.save(
            out_path,
            stats.astype(np.float32),
            allow_pickle=False,
    )
            
            
