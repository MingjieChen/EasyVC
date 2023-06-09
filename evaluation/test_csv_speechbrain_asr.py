import argparse
import csv
import json
import glob
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--eval_list', type = str)
    parser.add_argument('--eval_wav_dir', type = str) # absolute eval_wav_dir
    parser.add_argument('--test_csv_path', type = str)
    parser.add_argument('--n_splits', type  = int, default = 1)
    

    
    args = parser.parse_args()
    
    # load eval_list
    with open(args.eval_list) as f:
        eval_list = json.load(f)
        f.close()
    print(f'found {len(eval_list)} eval_list samples')

    # parser eval_wav_dir

    eval_wav_paths = sorted(glob.glob(os.path.join(args.eval_wav_dir, '*.wav')))
    print(f'found {len(eval_wav_paths)} samples')
    
    # check 
    assert len(eval_wav_paths) == len(eval_list), f'eval_wav_path {len(eval_wav_paths)} eval_list {len(eval_list)}'
    

    n_samples_per_split = len(eval_wav_paths) // args.n_splits
    print(f'n_samples_per_split {n_samples_per_split}')
    
    # write test_csv
    fieldnames = ["ID", "duration", "wav", "spk_id", "wrd"]
    
    for n in range(1, args.n_splits + 1):
        split_path = args.test_csv_path.split('.')[0] + f'.{n}.csv'
        #print(f'write csv path {split_path}')
        with open(split_path, 'w') as f:
            csv_writer = csv.DictWriter(f, fieldnames = fieldnames)
            csv_writer.writeheader()

            start = (n-1) * n_samples_per_split
            end = n * n_samples_per_split  
            if n == args.n_splits:
                split_eval_list = eval_list[start:]
            else:
                split_eval_list = eval_list[start:end]    
            for meta in split_eval_list:
                ID = meta['ID']
                wav_path = os.path.join(args.eval_wav_dir, ID + '_gen.wav')
                assert os.path.exists(wav_path)
                spk_id = ID.split('_')[-1]
                wrd = meta['text']
                duration = meta['duration']
                
                csv_writer.writerow({'ID': ID, 'duration': duration, 'wav': wav_path, 'spk_id':spk_id, 'wrd': wrd})
            f.close()    
                    
                    
                              

            

        
        

