import argparse
import csv
import os
import re


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    metadata_path = parser.add_argument('--metadata_path', type = str)
    out_path = parser.add_argument('--out_path', type = str)
    args = parser.parse_args()


    metadata = []

    with open(args.metadata_path) as f:
        csv_reader = csv.DictReader(f)
        csv_headers = csv_reader.fieldnames
        for row in csv_reader:
            metadata.append(row)
        f.close()
    
    updated_metadata = []
    csv_headers.append('wrd')
    for meta in metadata:
        wav_path = meta['wav_path']
        text_path = wav_path.replace('wav48','txt').replace('.wav', '.txt')
        assert os.path.exists(text_path)
        with open(text_path) as text_f:
            text = text_f.readlines()[0]
            assert type(text) == str
            text = text.upper()
            
            text = re.sub(r"[^A-Z ]", '', text)
        meta['wrd'] = text
        updated_metadata.append(meta)    
    
    
        
    with open(args.out_path, mode="w") as csv_f:
        csv_writer = csv.DictWriter(
            csv_f, fieldnames = csv_headers 
        )
        csv_writer.writeheader()

        for meta in updated_metadata:
            csv_writer.writerow(meta)
        csv_f.close()
            
