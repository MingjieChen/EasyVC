import glob
import argparse
import os
import sys
import re


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--split', type = str)
    parser.add_argument('--db', type = str)
    parser.add_argument('--data_dir', type = str)
    parser.add_argument('--spk', type = str)
    parser.add_argument('--num_dev', type = int, default = 10)
    args = parser.parse_args()


    scp_path = os.path.join(args.data_dir, args.split, 'wav.scp')
    text_path = os.path.join(args.data_dir, args.split, 'text')
    os.makedirs(os.path.dirname(scp_path), exist_ok = True)
    
    f = open(scp_path, 'a')
    f_text = open(text_path, 'a')
    if args.split == 'eval':
        wav_paths = sorted(glob.glob(os.path.join(args.db, args.spk, '[EGFM]3*.wav' )))
    else:
        wav_paths = list(sorted(glob.glob(os.path.join(args.db, args.spk, '[EGFM]1*.wav'))))\
                 + list(sorted(glob.glob(os.path.join(args.db, args.spk, '[EGFM]2*.wav'))))    
    if args.split == 'train_nodev':
        wav_paths = wav_paths[:-args.num_dev]             
    elif args.split == 'dev':
        wav_paths = wav_paths[-args.num_dev:]    

    for ind, wav_path in enumerate(wav_paths):
        basename = os.path.basename(wav_path).split('.')[0]
        if basename.startswith('E'):
            trans_path = os.path.join(args.db, 'prompts', 'Eng_transcriptions.txt')
        elif basename.startswith('G'):
            trans_path = os.path.join(args.db, 'prompts', 'Ger_transcriptions.txt')
        elif basename.startswith('F'):
            trans_path = os.path.join(args.db, 'prompts', 'Fin_transcriptions.txt')
        elif basename.startswith('M'):
            trans_path = os.path.join(args.db, 'prompts', 'Man_transcriptions.txt')

            
        id = f'{args.spk}_{basename}'
        for line in open(trans_path).readlines():
            line = line.strip()
            if line.split()[0] == basename[1:]:
                text = ' '.join(line.split()[1:])
                assert type(text) == str
                text = text.upper()
                if not basename.startswith('M'):
                    text = re.sub(r"[^A-Z ]", '', text)
        f.write(f'{id} {wav_path}\n')
        f_text.write(f'{id} {text}\n')
        print(f'vcc2020 {args.split}: ({id} {wav_path}) {text}')
    
    f.close()    

                
                    



    
