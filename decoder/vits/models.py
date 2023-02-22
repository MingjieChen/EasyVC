import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from .modules import LRELU_SLOPE, ResBlock1, ResBlock2, WN, ResidualCouplingLayer, Flip
from .attentions import Encoder

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from .commons import init_weights, get_padding, rand_slice_segments, sequence_mask


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
        
        out = x.transpose(1,2)
        out = self.pitch_convs(out)
        out = out.transpose(1,2)
        return out    

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
        return prosodic_reps.transpose(1, 2)     


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
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        #self.f0_emb = nn.Embedding(256, hidden_channels)

        self.enc_ =  Encoder(
                  hidden_channels,
                  filter_channels,
                  n_heads,
                  n_layers,
                  kernel_size,
                  p_dropout)

    def forward(self, x, x_lengths, pros=None):
        # x: ling_rep
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = x + pros
        x = self.enc_(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask

        return z, m, logs, x_mask

class ResidualCouplingBlock(nn.Module):
    def __init__(self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
            self.flows.append(Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        # x: z
        # g: spk_emb
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class PosteriorEncoder(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        # x: spec
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = ResBlock1 if resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
          x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2,3,5,7,11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class VITS(nn.Module):
    def __init__(self, config):
        
        super().__init__()

        self.generator = SynthesizerTrn(config)
        self.discriminator = MultiPeriodDiscriminator()
    def forward(self, x):
        pass    

class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(self, config):
    

        super().__init__()

    
        #self.n_vocab = n_vocab
        self.spec_channels = config['spec_channels']
        self.inter_channels = config['inter_channels']
        self.hidden_channels = config['hidden_channels']
        self.filter_channels = config['filter_channels']
        self.n_heads = config['n_heads']
        self.n_layers = config['n_layers']
        self.kernel_size = config['kernel_size']
        self.p_dropout = config['p_dropout']
        self.resblock = config['resblock']
        self.resblock_kernel_sizes = config['resblock_kernel_sizes']
        self.resblock_dilation_sizes = config['resblock_dilation_sizes']
        self.upsample_rates = config['upsample_rates']
        self.upsample_initial_channel = config['upsample_initial_channel']
        self.upsample_kernel_sizes = config['upsample_kernel_sizes']
        self.segment_size = config['segment_size']
        self.input_dim = config['input_dim']
        #self.use_sdp = use_sdp
        self.gin_channels = config['spk_emb_dim']
        self.hop_size = config['hop_length']
        
        if 'prosodic_rep_type' not in config:
            self.prosodic_net = None
        else:
            if config['prosodic_rep_type'] == 'discrete':     
                self.prosodic_net = DiscreteProsodicNet(config['prosodic_net'])
            elif config['prosodic_rep_type'] == 'continuous':
                self.prosodic_net = ContinuousProsodicNet(config['prosodic_net'])    
        self.enc_p = TextEncoder(self.input_dim, 
                        self.inter_channels, 
                        self.hidden_channels, 
                        5, 
                        1, 
                        16,
                        0, 
                        self.filter_channels, 
                        self.n_heads, 
                        self.p_dropout)

        self.dec = Generator(self.inter_channels, 
                            self.resblock, 
                            self.resblock_kernel_sizes, 
                            self.resblock_dilation_sizes, 
                            self.upsample_rates, 
                            self.upsample_initial_channel, 
                            self.upsample_kernel_sizes, 
                            gin_channels=self.gin_channels)
    
        self.enc_q = PosteriorEncoder(self.spec_channels, self.inter_channels, self.hidden_channels, 5, 1, 16, gin_channels=self.gin_channels)
        self.flow = ResidualCouplingBlock(self.inter_channels, self.hidden_channels, 5, 1, 4, gin_channels=self.gin_channels)



    def forward(self, spec, spec_lengths, ling, spk, pros):
        if self.prosodic_net is not None and pros is not None:
            pros = self.prosodic_net(pros)
        z_ptemp, m_p, logs_p, _ = self.enc_p(ling, spec_lengths, pros)

        z, m_q, logs_q, spec_mask = self.enc_q(spec, spec_lengths, g=spk)
        z_p = self.flow(z, spec_mask, g=spk)
        
        assert z.size(2) >= self.segment_size // self.hop_size, f'spec {spec.size()} ling {ling.size()} pros {pros.size()} z {z.size()}'

        z_slice, ids_slice = rand_slice_segments(z, spec_lengths, self.segment_size//self.hop_size)
        o = self.dec(z_slice, g=spk)
        return o, ids_slice, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def infer(self, ling, lengths, pros, spk):
        z_p, m_p, logs_p, mask = self.enc_p(ling, lengths, pros)
        z = self.flow(z_p, mask, g=spk, reverse=True)
        o = self.dec((z * mask), g=spk, pros = pros)
        return o 


