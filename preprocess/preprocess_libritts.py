import argparse
import os
import numpy  as np
from tqdm import tqdm
import csv
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import subprocess
import re


        

def process_speaker(spk_scp, spk, args):
    _metadata = []
    for scp_line, seg_line in spk_scp:
        # load wav
        ID = scp_line.split(' ')[0]
        wav_path = scp_line.split(' ')[1].strip()
        wav_path = os.path.join(args.data_root, wav_path)
        #text_path = wav_path.replace('.wav', '.normalized.txt')
        #assert os.path.exists(text_path)
        #with open(text_path) as text_f:
        #    text = text_f.readline().strip("\n")
        #    assert type(text) == str
        #    text = text.upper()
            
        #    text = re.sub(r"[^A-Z ]", '', text)
        #    text_f.close()
        
        
        # trim silence
        start, end = float(seg_line.split(' ')[2]), float(seg_line.split(' ')[3].strip())
        duration = end - start

        _metadata.append({"ID": ID, "wav_path": wav_path, "spk": spk, "start": start, "end": end, "duration": f'{duration:.2f}' })
    
    return _metadata    



    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', type = str)
    parser.add_argument('--scp_dir', type = str)
    parser.add_argument('--split', type = str)
    parser.add_argument('--metadata_dir', type = str)
    parser.add_argument('--max_workers', type = int, default = 20)
    args = parser.parse_args()


    # load scp and segments
    with open(os.path.join(args.scp_dir, args.split,'wav.scp')) as f:
        wav_scp = f.readlines()
        f.close()
    with open(os.path.join(args.scp_dir, args.split, 'segments')) as f:
        segments = f.readlines()
        f.close()
    
    # multi processer
    executor = ProcessPoolExecutor(max_workers=args.max_workers)
    futures = []
    metadata_headers = ['ID','wav_path', 'spk','start', 'end', 'duration']
    metadata = []
    #for spk in tqdm(speakers):
    with open(os.path.join(args.scp_dir, args.split,'wav.scp')) as f:
        wav_scp = f.readlines()
        f.close()
    with open(os.path.join(args.scp_dir, args.split, 'segments')) as f:
        segments = f.readlines()
        f.close()
    
    assert len(wav_scp) == len(segments), "wav_scp and segments length not equal"

    # build up a dict for spk2scp
    spk2scp = {}
    for _wav_scp, _segments in zip(wav_scp, segments):
        assert _wav_scp.split()[0] == _segments.split()[0]
        fid = _wav_scp.split()[0]
        spk = fid.split('_')[0]
        if spk not in spk2scp:
            spk2scp[spk] = []
        spk2scp[spk].append((_wav_scp, _segments))
    print({_spk: len(_scp) for _spk, _scp in spk2scp.items()})    
    speakers = sorted(spk2scp.keys())
    
    for spk in spk2scp:
        spk_scp = spk2scp[spk]        
    
        futures.append(executor.submit(partial(process_speaker, spk_scp, spk, args)))
    results = [future.result() for future in tqdm(futures)]    
    for res in results:
        metadata.extend(res)
    
    with open(os.path.join(args.metadata_dir, 'metadata.csv'), 'w') as f:
        writer = csv.DictWriter(f, fieldnames = metadata_headers)    
        writer.writeheader()
        for line in metadata:
           writer.writerow(line)
        f.close()
    with open(os.path.join(args.metadata_dir, 'speakers.txt'), 'w') as f:
        for spk in speakers:
            f.write(f"{spk}\n")
        f.close()
            
            
