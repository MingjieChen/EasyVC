# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .attentions import Encoder
from math import sqrt
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d

from tqdm import tqdm

Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d

def sequence_mask(length, max_length=None):
    if max_length is None:
      max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)
class TextEncoder(nn.Module):
    def __init__(self,
              in_channels,
              out_channels,
              hidden_channels,
              kernel_size,
              dilation_rate,
              n_layers,
              gin_channels=0,
              filter_channels=None,
              n_heads=None,
              p_dropout=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.proj = nn.Conv1d(hidden_channels, out_channels , 1)

        self.enc_ =  Encoder(
                  hidden_channels,
                  filter_channels,
                  n_heads,
                  n_layers,
                  kernel_size,
                  p_dropout)

    def forward(self, x, x_lengths):
        # x: ling_rep
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc_(x * x_mask, x_mask)
        out = self.proj(x) * x_mask

        return out, x_mask

def Conv1d(*args, **kwargs):
  layer = nn.Conv1d(*args, **kwargs)
  nn.init.kaiming_normal_(layer.weight)
  return layer


@torch.jit.script
def silu(x):
  return x * torch.sigmoid(x)

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

class DiffusionEmbedding(nn.Module):
  def __init__(self, max_steps):
    super().__init__()
    self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False)
    self.projection1 = Linear(128, 512)
    self.projection2 = Linear(512, 512)

  def forward(self, diffusion_step):
    if diffusion_step.dtype in [torch.int32, torch.int64]:
      x = self.embedding[diffusion_step]
    else:
      x = self._lerp_embedding(diffusion_step)
    x = self.projection1(x)
    x = silu(x)
    x = self.projection2(x)
    x = silu(x)
    return x

  def _lerp_embedding(self, t):
    low_idx = torch.floor(t).long()
    high_idx = torch.ceil(t).long()
    low = self.embedding[low_idx]
    high = self.embedding[high_idx]
    return low + (high - low) * (t - low_idx)

  def _build_embedding(self, max_steps):
    steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
    dims = torch.arange(64).unsqueeze(0)          # [1,64]
    table = steps * 10.0**(dims * 4.0 / 63.0)     # [T,64]
    table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
    return table


class Upsampler(nn.Module):
  def __init__(self, n_mels):
    super().__init__()
    # hard code upsampling scale to 240
    self.conv1 = ConvTranspose2d(1, 1, [3, 24], stride=[1, 12], padding=[1, 6])
    self.conv2 = ConvTranspose2d(1, 1,  [3, 40], stride=[1, 20], padding=[1, 10])

  def forward(self, x):
    x = torch.unsqueeze(x, 1)
    x = self.conv1(x)
    x = F.leaky_relu(x, 0.4)
    x = self.conv2(x)
    x = F.leaky_relu(x, 0.4)
    x = torch.squeeze(x, 1)
    return x


class ResidualBlock(nn.Module):
  def __init__(self, n_mels, residual_channels, dilation, spk_emb_dim):
    '''
    :param n_mels: inplanes of conv1x1 for spectrogram conditional
    :param residual_channels: audio conv
    :param dilation: audio conv dilation
    :param uncond: disable spectrogram conditional
    '''
    super().__init__()
    self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
    self.diffusion_projection = Linear(512, residual_channels)
    self.local_conditioner_projection = Conv1d(n_mels,  residual_channels, 1)
    self.global_conditioner_projection = Conv1d(spk_emb_dim, residual_channels, 1)

    self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

  def forward(self, x, diffusion_step, c, g):

    diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
    y = x + diffusion_step
    local_condition = self.local_conditioner_projection(c)
    global_condition = self.global_conditioner_projection(g)
    y = y + global_condition + local_condition
    y = self.dilated_conv(y) 
    
    gate, filter = torch.chunk(y, 2, dim=1)
    y = torch.sigmoid(gate) * torch.tanh(filter)

    y = self.output_projection(y)
    residual, skip = torch.chunk(y, 2, dim=1)
    return (x + residual) / sqrt(2.0), skip


class DiffWave(nn.Module):
  def __init__(self, config):
    super().__init__()
    
    residual_channels = config['residual_channels']
    input_dim = config['input_dim']
    inter_channels = config['inter_channels']
    hidden_channels = config['hidden_channels']
    filter_channels = config['filter_channels']
    n_heads = config['n_heads']
    p_dropout = config['p_dropout']
    kernel_size = config['kernel_size']
    n_layers = config['n_layers']

    residual_layers = config['residual_layers']
    dilation_cycle_length = config['dilation_cycle_length']
    self.spk_emb_dim = config['spk_emb_dim']
    
    self.use_text_encoder = config['use_text_encoder']

    
    noise_steps = config['noise_steps']
    noise_start = config['noise_start']
    noise_end = config['noise_end']   

    self.infer_noise = config['infer_noise']


    noise_schedule = np.linspace(noise_start, noise_end, noise_steps).tolist()
    self.noise_schedule = noise_schedule
    self.diffusion_embedding = DiffusionEmbedding(len(noise_schedule))
    self.fast_sampling  = config['fast_sampling'] if 'fast_sampling' in config else True
    self.upsampler = Upsampler(inter_channels)
    
    if self.use_text_encoder:
        self.text_encoder = TextEncoder(
                        input_dim, 
                        inter_channels, 
                        hidden_channels, 
                        kernel_size, 
                        1, 
                        n_layers,
                        0, 
                        filter_channels, 
                        n_heads, 
                        p_dropout)
    else:
        self.text_encoder = nn.Conv1d(input_dim, inter_channels, 3,1,1)        
    
    if 'prosodic_rep_type' not in config:
        self.prosodic_net = None
    else:
        if config['prosodic_rep_type'] == 'discrete':     
            self.prosodic_net = DiscreteProsodicNet(config['prosodic_net'])
        elif config['prosodic_rep_type'] == 'continuous':
            self.prosodic_net = ContinuousProsodicNet(config['prosodic_net'])    
    
    #self.reduce_proj = nn.Conv1d(self.spk_emb_dim + inter_channels, inter_channels, 1,1,0)

    self.input_projection = Conv1d(1, residual_channels, 1)
    self.residual_layers = nn.ModuleList([
        ResidualBlock(inter_channels, residual_channels, 2**(i % dilation_cycle_length), spk_emb_dim = self.spk_emb_dim)
        for i in range(residual_layers)
    ])

    self.skip_projection = Conv1d(residual_channels, residual_channels, 1)
    self.output_projection = Conv1d(residual_channels, 1, 1)
    nn.init.zeros_(self.output_projection.weight)

  def forward(self, audio, diffusion_step, ling, pros, spk, lengths):
    
    
    if self.use_text_encoder:
        x, x_masks = self.text_encoder(ling, lengths)
    else:
        x = self.text_encoder(ling)    

    if self.prosodic_net is not None and pros is not None:
        pros = self.prosodic_net(pros)
        x += pros
    
    up_spk_embeds = F.normalize(
            spk.squeeze(2)).unsqueeze(2).expand(ling.size(0), self.spk_emb_dim, ling.size(2) * 240)
        
    x = self.upsampler(x)

    y = audio.unsqueeze(1)
    y = self.input_projection(y)
    y = F.relu(y)

    diffusion_step = self.diffusion_embedding(diffusion_step)

    skip = None
    for layer in self.residual_layers:
      y, skip_connection = layer(y, diffusion_step, x, up_spk_embeds)
      skip = skip_connection if skip is None else skip_connection + skip

    y = skip / sqrt(len(self.residual_layers))
    y = self.skip_projection(y)
    y = F.relu(y)
    y = self.output_projection(y)
    return y
  
  def inference(self, ling, pros, spk, lengths):
    fast_sampling = self.fast_sampling
    training_noise_schedule = np.array(self.noise_schedule)    
    #inference_noise_schedule= np.array([0.0001, 0.001, 0.01, 0.1, 0.2, 0.5])
    inference_noise_schedule = np.array(self.infer_noise)
    inference_noise_schedule = np.array(inference_noise_schedule) if fast_sampling else training_noise_schedule
    print(f'inference noise schedule {inference_noise_schedule}')  
    talpha = 1 - training_noise_schedule
    talpha_cum = np.cumprod(talpha)

    beta = inference_noise_schedule
    alpha = 1 - beta
    alpha_cum = np.cumprod(alpha)

    T = []
    for s in range(len(inference_noise_schedule)):
      for t in range(len(training_noise_schedule) - 1):
        if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
          twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
          T.append(t + twiddle)
          break
    T = np.array(T, dtype=np.float32)
    print(f'inference T {T}')
    
    # hard code hop_size = 240
    audio = torch.randn(ling.shape[0], 240 * ling.shape[-1], device=ling.device)
    noise_scale = torch.from_numpy(alpha_cum**0.5).float().unsqueeze(1).to(ling.device)
    for n in tqdm(range(len(alpha) - 1, -1, -1)):
      c1 = 1 / alpha[n]**0.5
      c2 = beta[n] / (1 - alpha_cum[n])**0.5
      audio = c1 * (audio - c2 * self.forward(audio, torch.tensor([T[n]], device=audio.device), ling, pros, spk, lengths).squeeze(1))
      if n > 0:
        noise = torch.randn_like(audio)
        sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
        audio += sigma * noise
      print(f'c1 {c1} c2 {c2} sigma {sigma}', flush = True)
      audio = torch.clamp(audio, -1.0, 1.0)
    return audio 
    
    
