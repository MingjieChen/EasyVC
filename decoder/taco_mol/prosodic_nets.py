import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import torch.nn.functional as F
class DiscreteProsodicNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        n_bins = config['prosodic_bins']
        prosodic_stats_path = config['prosodic_stats_path']
        hidden_dim = config['hidden_dim']
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
        self.pitch_convs = torch.nn.Sequential(
            torch.nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            torch.nn.LeakyReLU(0.1),

            torch.nn.Conv1d(
                hidden_dim, hidden_dim, 
                kernel_size= 3, 
                stride=2, 
                padding=1,
            ),
            torch.nn.LeakyReLU(0.1),
            
            torch.nn.Conv1d(
                hidden_dim, hidden_dim, 
                kernel_size= 3, 
                stride=2, 
                padding=1,
            ),
            torch.nn.LeakyReLU(0.1),

        )
    def forward(self, x):
        pitch = x[:,:,0]
        energy = x[:,:,1]
        pitch_reps = self.pitch_embedding(torch.bucketize(pitch, self.pitch_bins))
        energy_reps = self.energy_embedding(torch.bucketize(energy, self.energy_bins))
        prosodic_reps = pitch_reps + energy_reps
        out = prosodic_reps.transpose(1,2)
        out = self.pitch_convs(out)
        out = out.transpose(1,2)
        return prosodic_reps     
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
                stride=2, 
                padding=1,
            ),
            torch.nn.LeakyReLU(0.1),
            
            torch.nn.InstanceNorm1d(hidden_dim, affine=False),
            torch.nn.Conv1d(
                hidden_dim, hidden_dim, 
                kernel_size= 3, 
                stride=2, 
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

