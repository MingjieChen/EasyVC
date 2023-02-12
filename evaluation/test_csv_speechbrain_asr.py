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
    
    
    args = parser.parse_args()
    
    # load eval_list
    with open(args.eval_list) as f:
        eval_list = json.load(f)
        f.close()
    
    # parser eval_wav_dir

    eval_wav_paths = sorted(glob.glob(os.path.join(args.eval_wav_dir, '*.wav')))
    
    # check 
    assert len(eval_wav_paths) == len(eval_list)
    
    # write test_csv
    fieldnames = ["ID", "duration", "wav", "spk_id", "wrd"]
    with open(args.test_csv_path, 'w') as f:
        csv_writer = csv.DictWriter(f, fieldnames = fieldnames)
        csv_writer.writeheader()

        for meta in eval_list:
            ID = meta['ID']
            wav_path = os.path.join(args.eval_wav_dir, ID + '_gen.wav')
            assert os.path.exists(wav_path)
            spk_id = ID.split('_')[-1]
            wrd = meta['wrd']
            duration = meta['duration']
            
            csv_writer.writerow({'ID': ID, 'duration': duration, 'wav': wav_path, 'spk_id':spk_id, 'wrd': wrd})
            

        
        

