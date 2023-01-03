import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_mask_from_lengths
from module import MelDecoder, PhonemeEncoder, VarianceAdaptor
import math
from torch.nn.utils import spectral_norm
class Generator(nn.Module):
    ''' a TTS model based on FastSpeech2 '''
    def __init__(self, config):
        
        super().__init__()
        # model
        self.phoneme_encoder = PhonemeEncoder(config)
        #self.variance_adaptor = VarianceAdaptor(config)
        self.mel_decoder = MelDecoder(config)
        if 'use_spk_embs' in config and config['use_spk_embs'] :
            self.speaker_embedding = None
        else:    
            self.speaker_embedding = nn.Embedding(config['num_speakers'], config['spk_emb_dim']) 
        self.mel_linear = nn.Linear(config['transformer']['decoder_hidden'],80)
    
    def forward(self, ling_feat, speakers, length, max_len):    
        
        mask = get_mask_from_lengths(length, max_len)
        if self.speaker_embedding is not None:
            speaker_embedding = self.speaker_embedding(speakers)
        else:
            speaker_embedding = speakers    
        out, _ = self.phoneme_encoder(ling_feat, speaker_embedding, mask )
        #out = self.variance_adaptor(out, speaker_embedding, mask)
        out_mel, mel_mask = self.mel_decoder(out, speaker_embedding, mask)
        out_mel = self.mel_linear(out_mel)
        return out_mel, mel_mask
class Generator1(nn.Module):
    ''' a TTS model based on FastSpeech2 '''
    def __init__(self, config):
        
        super().__init__()
        # model
        #self.phoneme_encoder = PhonemeEncoder(config)
        #self.variance_adaptor = VarianceAdaptor(config)
        self.mel_decoder = MelDecoder(config)
        if 'use_spk_embs' in config and config['use_spk_embs'] :
            self.speaker_embedding = None
        else:    
            self.speaker_embedding = nn.Embedding(config['num_speakers'], config['spk_emb_dim']) 
        self.mel_linear = nn.Linear(config['transformer']['decoder_hidden'],80)
    
    def forward(self, ling_feat, speakers, length, max_len):    
        
        mask = get_mask_from_lengths(length, max_len)
        if self.speaker_embedding is not None:
            speaker_embedding = self.speaker_embedding(speakers)
        else:
            speaker_embedding = speakers    
        #out, _ = self.phoneme_encoder(ling_feat, speaker_embedding, mask )
        #out = self.variance_adaptor(out, speaker_embedding, mask)
        out_mel, mel_mask = self.mel_decoder(ling_feat, speaker_embedding, mask)
        out_mel = self.mel_linear(out_mel)
        
        return out_mel, mel_mask


class DisRes(nn.Module):
    
    """Residual block in Discriminator"""
    def __init__(self, dim_in, dim_out, ):
        super().__init__()
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3,1,1)
        #self.conv1 = spectral_norm(nn.Conv2d(dim_in, dim_in, 3,1,1))
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3,1,1)
        #self.conv2 = spectral_norm(nn.Conv2d(dim_in, dim_out, 3,1,1))
        if dim_in != dim_out:
            self.sc_conv = True
            #self.conv1x1 = spectral_norm(nn.Conv2d(dim_in, dim_out, 1,1,0, bias = False))
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1,1,0, bias = False)
        else:
            self.sc_conv = False
        self.act = nn.LeakyReLU(0.2)
    def forward(self,x ):
        if self.sc_conv:
            residual = self.conv1x1(x)
        else:
            residual = x
        residual = F.avg_pool2d(residual, 2)
        
        x = self.act(x)
        x = self.conv1(x)
        x = F.avg_pool2d(x,2)
        x = self.act(x)
        x = self.conv2(x)
        

        out = x + residual

        out =  out / math.sqrt(2)
        return out
class SpeakerDiscriminator(nn.Module):
    def __init__(self, config):
        super(SpeakerDiscriminator, self).__init__()

        num_speakers = config['speaker_discriminator']['num_speakers']
        self.max_seq_len = config['speaker_discriminator']['max_seq_len']
        
        # Initial layers.
        
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
        )
        self.speaker_embs = nn.Embedding(num_speakers, 256)

        self.down_sample_1 = DisRes(64, 128)
        self.down_sample_2 = DisRes(128, 128)
        self.down_sample_3 = DisRes(128, 128)
        self.down_sample_4 = DisRes(128, 128)
        
        
        blocks = []
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(128,128, 5,1,0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        self.blocks = nn.Sequential(*blocks)            
        #self.dis_conv = nn.Conv2d(256, num_speakers, kernel_size = 1, stride = 1, padding = 0, bias = False )
        self.V = nn.Linear(256,128)
        #self.project = nn.Linear(128,1)
        self.w_b_0 = nn.Linear(1,1)
    def forward(self, x, spk_id, mask, max_len):
        max_len = min(max_len, self.max_seq_len)
        x = x[:, :max_len, :]
        x = x.masked_fill(mask.unsqueeze(-1)[:,:max_len:],0)
        # x: [B,T,C]
        x = x.transpose(1,2).unsqueeze(1) # convert to shape [B,1,C,T]
        x = self.conv_layer_1(x) 

        x = self.down_sample_1(x)
        
        
        x = self.down_sample_2(x)
        
        x = self.down_sample_3(x)
        
        x = self.down_sample_4(x)
        
        x = self.blocks(x)
        b, c, h, w = x.size()
        x = x.view(b,c)
        
        speakers = self.speaker_embs(spk_id.squeeze(1))

        V = self.V(speakers)

        out = torch.sum(x*V , dim = 1, keepdim= True)
        out =  self.w_b_0(out).squeeze()



        return out
