import argparse
import os
import sys
import glob
from speechbrain.utils.metric_stats import ErrorRateStats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--decoding_output_dirs', type = str, nargs = '+')
    parser.add_argument('--decoding_output_filename', type = str, default = 'token.txt')
    parser.add_argument('--wer_output_path', type = str)
    
    args = parser.parse_args()
    # define wer_metric
    wer_metric = ErrorRateStats()

    # find all decoding output files
    assert len(args.decoding_output_dirs) != 0
    print(f'found {len(args.decoding_output_dirs)} decoding output dirs')
    n_decoding_outputs = len(args.decoding_output_dirs)
    # find all output files
    for output_dir in args.decoding_output_dirs:
        output_file_path = os.path.join(output_dir, args.decoding_output_filename)
        assert os.path.exists(output_file_path), f'{output_file_path} does not exist'
        with open(output_file_path) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                id, hyp, ref = line.split('|')
                wer_metric.append([id], [hyp], [ref])
            f.close()    
    WER = wer_metric.summarize('error_rate')
    print(f'WER {WER}')
    with open(args.wer_output_path, 'w') as  f:
        wer_metric.write_stats(f)
        f.close()
    print(f'see wer results in {args.wer_output_path}')    
        
               
    



