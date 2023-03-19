import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_mask_from_lengths
from .module import Decoder, Encoder, VarianceAdaptor, PostNet
import math
from torch.nn.utils import spectral_norm
class FastSpeech2(nn.Module):
    ''' FastSpeech2 '''
    def __init__(self, config):
        
        super().__init__()
        # model
        self.use_text_encoder = config['use_text_encoder']
        if self.use_text_encoder:

            self.encoder = Encoder(config)
        else:
            self.encoder = nn.Conv1d(config['input_dim'], config['transformer']['decoder_hidden'], 3, 1, 1)    
        self.variance_adaptor = VarianceAdaptor(config)
        self.decoder = Decoder(config)
        self.mel_linear = nn.Linear(config['transformer']['decoder_hidden'], config['out_dim'])
        if config['postnet']['n_layers'] != 0:
            self.postnet = PostNet(**config['postnet'])
        else:
            self.postnet = None    
    
    def forward(self, ling_rep, pros_rep, spk_emb, length, max_len):    
        
        mask = get_mask_from_lengths(length, max_len)
        if self.use_text_encoder:

            out, _ = self.encoder(ling_rep, mask)
        else:
            out = self.encoder(ling_rep.transpose(1,2)).transpose(1,2)    
        out, _ = self.variance_adaptor(out, spk_emb, pros_rep, mask, max_len)
        out_mel, mel_mask = self.decoder(out, mask)
        out_mel = self.mel_linear(out_mel)
        if self.postnet is not None:
            postnet_mel = self.postnet(out_mel) + out_mel
        else:
            postnet_mel = None    
            

        return out_mel, postnet_mel, mel_mask
