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

    args = parser.parse_args()
    # create scaler
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
        mel_path = os.path.join(args.dump_dir, args.split, 'mel', spk, ID+'.npy')
        assert os.path.exists(mel_path), f'{mel_path}'
        mel = np.load(mel_path)
        scaler.partial_fit(mel)
    
    out_path = os.path.join(args.dump_dir, args.split, 'mel', args.split + '.npy')    
    stats = np.stack([scaler.mean_, scaler.scale_], axis=0)
    np.save(
            out_path,
            stats.astype(np.float32),
            allow_pickle=False,
    )
            
            
