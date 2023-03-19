# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model.base import BaseModule
from .model.text_encoder import TextEncoder
from .model.diffusion import Diffusion
from .model.utils import sequence_mask, generate_path, duration_loss, fix_len_compatibility

import numpy as np



class DiscreteProsodicNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        n_bins = config['prosodic_bins']
        prosodic_stats_path = config['prosodic_stats_path']
        # load pitch energy min max
        stats = np.load(prosodic_stats_path)
        pitch_max = stats[0][0]
        pitch_min = stats[1][0]
        energy_max = stats[2][0]
        energy_min = stats[3][0]
        self.pitch_bins = nn.Parameter(
                torch.linspace(pitch_min, pitch_max, n_bins - 1),
                requires_grad=False,
                )
        self.energy_bins = nn.Parameter(
                torch.linspace(energy_min, energy_max, n_bins - 1),
                requires_grad=False,
                )        
        self.pitch_embedding = nn.Embedding(
                n_bins, config["hidden_dim"]
                )
        self.energy_embedding = nn.Embedding(
                n_bins, config["hidden_dim"]
                )
    def forward(self, x):
        pitch = x[:,0,:]
        energy = x[:,1,:]
        pitch_reps = self.pitch_embedding(torch.bucketize(pitch, self.pitch_bins))
        energy_reps = self.energy_embedding(torch.bucketize(energy, self.energy_bins))
        prosodic_reps = pitch_reps + energy_reps
        return prosodic_reps.transpose(1,2)     
class ContinuousProsodicNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        hidden_dim = config['hidden_dim']
        self.pitch_convs = torch.nn.Sequential(
            torch.nn.Conv1d(2, hidden_dim, kernel_size=1, bias=False),
            torch.nn.LeakyReLU(0.1),

            torch.nn.InstanceNorm1d(hidden_dim, affine=False),
            torch.nn.Conv1d(
                hidden_dim, hidden_dim, 
                kernel_size= 3, 
                stride=1, 
                padding=1,
            ),
            torch.nn.LeakyReLU(0.1),
            
            torch.nn.InstanceNorm1d(hidden_dim, affine=False),
            torch.nn.Conv1d(
                hidden_dim, hidden_dim, 
                kernel_size= 3, 
                stride=1, 
                padding=1,
            ),
            torch.nn.LeakyReLU(0.1),

            torch.nn.InstanceNorm1d(hidden_dim, affine=False),
        )
    def forward(self, x):
        
        out = self.pitch_convs(x)
        return out    
class GradTTS(BaseModule):
    def __init__(self, config):
        super(GradTTS, self).__init__()

        #self.n_vocab = n_vocab
        self.input_dim = config['input_dim']
        #self.n_spks = n_spks
        self.spk_emb_dim = config['spk_emb_dim']
        self.n_enc_channels = config['n_enc_channels']
        self.filter_channels = config['filter_channels']
        self.filter_channels_dp = config['filter_channels_dp']
        self.n_heads = config['n_heads']
        self.n_enc_layers = config['n_enc_layers']
        self.enc_kernel = config['enc_kernel']
        self.enc_dropout = config['enc_dropout']
        self.window_size = config['window_size']
        self.n_feats = config['n_feats']
        self.dec_dim = config['dec_dim']
        self.beta_min = config['beta_min']
        self.beta_max = config['beta_max']
        self.pe_scale = config['pe_scale']
        self.use_prior_loss = config['use_prior_loss']
        
        self.use_text_encoder = config['use_text_encoder']
        if self.use_text_encoder:
            self.encoder = TextEncoder(self.input_dim, 
                                        self.n_feats, 
                                        self.n_enc_channels, 
                                        self.filter_channels, 
                                        self.filter_channels_dp, 
                                        self.n_heads, 
                                        self.n_enc_layers, 
                                        self.enc_kernel, 
                                        self.enc_dropout, 
                                        self.window_size)
        else:
            self.encoder = nn.Conv1d(self.input_dim, self.n_feats, 3,1,1)

        self.decoder = Diffusion(self.n_feats, self.dec_dim, self.beta_min, self.beta_max, self.pe_scale)

        if 'prosodic_rep_type' not in config:
            self.prosodic_net = None
        elif config['prosodic_rep_type'] == 'discrete':
            self.prosodic_net = DiscreteProsodicNet(config['prosodic_net'])
        elif config['prosodic_rep_type'] == 'continuous':    
            self.prosodic_net = ContinuousProsodicNet(config['prosodic_net'])
        else:
            raise Exception    
        # speaker embedding integration    
        self.reduce_proj = torch.nn.Conv1d(self.n_feats + self.spk_emb_dim, self.n_feats, 1,1,0)

    @torch.no_grad()
    def forward(self, ling, ling_lengths, spk, pros, n_timesteps, temperature=1.0, stoc=False, length_scale=1.0):
        
        if self.use_text_encoder:
            mu_x, x_mask = self.encoder(ling, ling_lengths)
        else:
            mu_x = self.encoder(ling)    

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        y_max_length = int(ling_lengths.max())
        y_max_length_ = fix_len_compatibility(y_max_length)
        
        
        # integrate prosodic representation
        if self.prosodic_net is not None and pros is not None:
            mu_x = mu_x + self.prosodic_net(pros)
        
        # integrate speaker representation
        spk_embeds = F.normalize(
                spk.squeeze(1)).unsqueeze(2).expand(ling.size(0), self.spk_emb_dim, y_max_length)
        mu_x = torch.cat([mu_x, spk_embeds], dim=1)
        mu_x = self.reduce_proj(mu_x)



        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(ling_lengths, y_max_length_).unsqueeze(1).to(ling.dtype)
        
        
        #if y_max_length_ > y_max_length:
        #    mu_x = torch.nn.functional.pad(mu_x, (0, y_max_length_ - y_max_length))


        # Sample latent representation from terminal distribution N(mu_y, I)
        z = mu_x + torch.randn_like(mu_x, device=mu_x.device) / temperature
        # Generate sample by performing reverse dynamics
        decoder_outputs = self.decoder(z, y_mask, mu_x, n_timesteps, stoc)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]

        return decoder_outputs 

    def compute_loss(self, ling, ling_lengths, mel, mel_lengths, spk, pros, out_size=None):
        # input dim: [B,C,T]
        
        if self.use_text_encoder:
            mu_x, x_mask = self.encoder(ling, ling_lengths)
        else:
            mu_x = self.encoder(ling)    
        
        
        # integrate prosodic representation
        if self.prosodic_net is not None and pros is not None:
            mu_x = mu_x + self.prosodic_net(pros)
        
        # integrate speaker representation
        spk_embeds = F.normalize(
                spk.squeeze(1)).unsqueeze(2).expand(ling.size(0), self.spk_emb_dim, ling.size(2))
        mu_x = torch.cat([mu_x, spk_embeds], dim=1)
        
        mu_x = self.reduce_proj(mu_x)

        mel_max_length = mel.shape[-1]
        _mel_max_length = fix_len_compatibility(mel_max_length) 
        mel_mask = sequence_mask(mel_lengths, _mel_max_length).unsqueeze(1).to(ling.dtype)
        
        # pad mu_x
        if _mel_max_length > mel_max_length:
            mu_x = torch.nn.functional.pad(mu_x, (0, _mel_max_length - mel_max_length))
            mel = torch.nn.functional.pad(mel, (0, _mel_max_length - mel_max_length))


        # Compute loss of score-based decoder
        diff_loss, xt = self.decoder.compute_loss(mel, mel_mask, mu_x)
        
        if self.use_prior_loss:
        # Compute loss between aligned encoder outputs and mel-spectrogram
            prior_loss = torch.sum(0.5 * ((mel - mu_x) ** 2 + math.log(2 * math.pi)) * mel_mask)
            prior_loss = prior_loss / (torch.sum(mel_mask) * self.n_feats)
            loss = diff_loss + prior_loss
            return loss, {'diff_loss': diff_loss.item(), 'prior_loss': prior_loss.item()}
        else:
            loss = diff_loss
        
            return loss, {'diff_loss': loss.item()}
