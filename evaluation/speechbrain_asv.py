from speechbrain.pretrained import SpeakerRecognition
import os
from speechbrain.utils.metric_stats import EER
import sys
from tqdm import tqdm

if __name__ == '__main__':

    converted_wav_dir = sys.argv[1]
    positive_pairs_path = sys.argv[2]
    negative_pairs_path = sys.argv[3]
    device = 'cuda'   


    verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb", run_opts={"device":f"{device}"})
    print(f'load verification model done')

    
    positive_pairs = []
    with open(positive_pairs_path) as f:
        for line in f:
            ID, wav = line.split()[0], line.split()[1].strip()
            positive_pairs.append((ID, wav))
        f.close()    
    
    negative_pairs = []
    with open(negative_pairs_path) as f:
        for line in f:
            ID, wav = line.split()[0], line.split()[1].strip()
            negative_pairs.append((ID, wav))
        f.close()    
    
    positive_scores = []
    for pair in tqdm(positive_pairs, total = len(positive_pairs)):
        ID, wav = pair
        wav_1 = os.path.join(converted_wav_dir, ID + '_gen.wav')
        wav_2 = wav

        score, prediction = verification.verify_files(wav_1, wav_2) 
        positive_scores.append(score)

    negative_scores = []
    for pair in tqdm(negative_pairs, total = len(negative_pairs)):
        ID, wav = pair
        wav_1 = os.path.join(converted_wav_dir, ID + '_gen.wav')
        wav_2 = wav

        score, prediction = verification.verify_files(wav_1, wav_2) 
        negative_scores.append(score)
    
    positive_scores = torch.tensor(positive_scores)     
    negative_scores = torch.tensor(negative_scores)     
    eer_result = EER(positive_scores, negative_scores)
    print(f'EER result {eer_result * 100}')
