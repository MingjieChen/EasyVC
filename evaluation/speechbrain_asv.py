from speechbrain.pretrained import EncoderClassifier
import os
from speechbrain.utils.metric_stats import EER
import sys
from tqdm import tqdm
import torch

class SpeakerRecognition(EncoderClassifier):
    """A ready-to-use model for speaker recognition. It can be used to
    perform speaker verification with verify_batch().

    ```
    Example
    -------
    >>> import torchaudio
    >>> from speechbrain.pretrained import SpeakerRecognition
    >>> # Model is downloaded from the speechbrain HuggingFace repo
    >>> tmpdir = getfixture("tmpdir")
    >>> verification = SpeakerRecognition.from_hparams(
    ...     source="speechbrain/spkrec-ecapa-voxceleb",
    ...     savedir=tmpdir,
    ... )

    >>> # Perform verification
    >>> signal, fs = torchaudio.load("tests/samples/single-mic/example1.wav")
    >>> signal2, fs = torchaudio.load("tests/samples/single-mic/example2.flac")
    >>> score, prediction = verification.verify_batch(signal, signal2)
    """

    MODULES_NEEDED = [
        "compute_features",
        "mean_var_norm",
        "embedding_model",
        "mean_var_norm_emb",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    def verify_batch(
        self, wavs1, wavs2, wav1_lens=None, wav2_lens=None, threshold=0.25
    ):
        """Performs speaker verification with cosine distance.

        It returns the score and the decision (0 different speakers,
        1 same speakers).

        Arguments
        ---------
        wavs1 : Torch.Tensor
                Tensor containing the speech waveform1 (batch, time).
                Make sure the sample rate is fs=16000 Hz.
        wavs2 : Torch.Tensor
                Tensor containing the speech waveform2 (batch, time).
                Make sure the sample rate is fs=16000 Hz.
        wav1_lens: Torch.Tensor
                Tensor containing the relative length for each sentence
                in the length (e.g., [0.8 0.6 1.0])
        wav2_lens: Torch.Tensor
                Tensor containing the relative length for each sentence
                in the length (e.g., [0.8 0.6 1.0])
        threshold: Float
                Threshold applied to the cosine distance to decide if the
                speaker is different (0) or the same (1).

        Returns
        -------
        score
            The score associated to the binary verification output
            (cosine distance).
        prediction
            The prediction is 1 if the two signals in input are from the same
            speaker and 0 otherwise.
        """
        emb1 = self.encode_batch(wavs1, wav1_lens, normalize=True)
        emb2 = self.encode_batch(wavs2, wav2_lens, normalize=True)
        score = self.similarity(emb1, emb2)
        return score, score > threshold

    def verify_files(self, path_x, path_y):
        """Speaker verification with cosine distance

        Returns the score and the decision (0 different speakers,
        1 same speakers).

        Returns
        -------
        score
            The score associated to the binary verification output
            (cosine distance).
        prediction
            The prediction is 1 if the two signals in input are from the same
            speaker and 0 otherwise.
        """
        waveform_x = self.load_audio(path_x, savedir = "pretrained_models/spkrec-ecapa-voxceleb") 
        waveform_y = self.load_audio(path_y, savedir = "pretrained_models/spkrec-ecapa-voxceleb")
        # Fake batches:
        batch_x = waveform_x.unsqueeze(0)
        batch_y = waveform_y.unsqueeze(0)
        # Verify:
        score, decision = self.verify_batch(batch_x, batch_y)
        # Squeeze:
        return score[0], decision[0]
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
    eer_result = EER(positive_scores, negative_scores)[0]
    print(f'EER result {eer_result * 100 }')
