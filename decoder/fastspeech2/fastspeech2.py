import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_mask_from_lengths
from .module import Decoder, Encoder, VarianceAdaptor
import math
from torch.nn.utils import spectral_norm
class FastSpeech2(nn.Module):
    ''' FastSpeech2 '''
    def __init__(self, config):
        
        super().__init__()
        # model
        self.encoder = Encoder(config)
        self.variance_adaptor = VarianceAdaptor(config)
        self.decoder = Decoder(config)
        self.mel_linear = nn.Linear(config['transformer']['decoder_hidden'],80)
    
    def forward(self, ling_rep, pros_rep, spk_emb, length, max_len):    
        
        mask = get_mask_from_lengths(length, max_len)
        out, _ = self.encoder(ling_rep, mask)
        out, _ = self.variance_adaptor(out, spk_emb, pros_rep, mask, max_len)
        out_mel, mel_mask = self.decoder(out, mask)
        out_mel = self.mel_linear(out_mel)
        return out_mel, mel_mask
