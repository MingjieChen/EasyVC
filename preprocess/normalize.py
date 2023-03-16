import argparse
import os

import numpy as np
import csv

from sklearn.preprocessing import StandardScaler
from tqdm import tqdm



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--stats_path', type = str)
    parser.add_argument('--dump_dir', type = str)
    parser.add_argument('--metadata', type = str)
    parser.add_argument('--split', type = str)
    parser.add_argument('--feature_type', type = str, default = 'mel', choices = ['mel', 'fastspeech2_pitch_energy', 'bigvgan_mel'])

    args = parser.parse_args()
    

    metadata = []
    # load metadata
    with open(args.metadata) as f:
        csv_reader = csv.DictReader(f)
        for meta in csv_reader:
            metadata.append(meta)
        f.close()
        
        
    # load stats
    if args.feature_type == 'fastspeech2_pitch_energy':
        scaler_pitch = StandardScaler()
        scaler_energy = StandardScaler()
        stats = np.load(args.stats_path)
        scaler_pitch.mean_ = stats[0]
        scaler_pitch.scale_ = stats[1]
        scaler_energy.mean_ = stats[2]
        scaler_energy.scale_ = stats[3]
        scaler_pitch.n_features_in_ = scaler_pitch.mean_.shape[0]
        scaler_energy.n_features_in_ = scaler_energy.mean_.shape[0]
        max_value_pitch = np.finfo(np.float64).min
        min_value_pitch = np.finfo(np.float64).max
        max_value_energy = np.finfo(np.float64).min
        min_value_energy = np.finfo(np.float64).max
    else:    

        scaler = StandardScaler()
        scaler.mean_ = np.load(args.stats_path)[0]
        scaler.scale_ = np.load(args.stats_path)[1]        
        scaler.n_features_in_ = scaler.mean_.shape[0]

    
        
    
    # normalize
    for _meta in tqdm(metadata, total = len(metadata)):
        ID = _meta['ID']
        spk = _meta['spk']
        feature_path = os.path.join(args.dump_dir, args.split, args.feature_type, spk, ID + '.npy')
        norm_path = os.path.join(args.dump_dir, args.split, f'norm_{args.feature_type}', spk, ID + '.npy')
        os.makedirs(os.path.dirname(norm_path), exist_ok = True)

        feature = np.load(feature_path)
        if args.feature_type == 'fastspeech2_pitch_energy':
            pitch_feature = feature[0,:]
            energy_feature = feature[1,:]
            norm_pitch = scaler_pitch.transform(pitch_feature.reshape(-1, 1))
            norm_energy = scaler_energy.transform(energy_feature.reshape(-1, 1))
            
            np.save(
                    norm_path,
                    np.array([norm_pitch.reshape(-1),norm_energy.reshape(-1)]),
                    allow_pickle=False,
            )
            if args.split.startswith('train_nodev'):
                max_value_pitch = max(max_value_pitch, max(norm_pitch))
                min_value_pitch = min(min_value_pitch, min(norm_pitch))
                max_value_energy = max(max_value_energy, max(norm_energy))
                min_value_energy = min(min_value_energy, min(norm_energy))
                min_max_stats = np.stack([
                        max_value_pitch,
                        min_value_pitch,
                        max_value_energy,
                        min_value_energy], axis = 0)
                np.save(os.path.join(os.path.dirname(args.stats_path), 'pitch_energy_min_max.npy'), min_max_stats, allow_pickle = False)
        else:    
            norm_feature = scaler.transform(feature)
            np.save(
                    norm_path,
                    norm_feature.astype(np.float32),
                    allow_pickle=False,
                   )




            


