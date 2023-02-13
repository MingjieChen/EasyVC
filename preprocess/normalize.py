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

    args = parser.parse_args()
    

    metadata = []
    # load metadata
    with open(args.metadata) as f:
        csv_reader = csv.DictReader(f)
        for meta in csv_reader:
            metadata.append(meta)
        f.close()
        
        
    # load stats
    scaler = StandardScaler()
    scaler.mean_ = np.load(args.stats_path)[0]
    scaler.scale_ = np.load(args.stats_path)[1]        
    scaler.n_features_in_ = scaler.mean_.shape[0]


    # normalize
    for _meta in tqdm(metadata):
        ID = _meta['ID']
        spk = _meta['spk']
        mel_path = os.path.join(args.dump_dir, args.split, 'mel', spk, ID + '.npy')
        norm_path = os.path.join(args.dump_dir, args.split, 'norm_mel', spk, ID + '.npy')
        os.makedirs(os.path.dirname(norm_path), exist_ok = True)

        mel = np.load(mel_path)
        norm_mel = scaler.transform(mel)
        np.save(
                norm_path,
                norm_mel.astype(np.float32),
                allow_pickle=False,
               )




            


