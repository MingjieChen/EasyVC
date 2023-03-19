import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import torch.nn.functional as F
import six
from .blocks import (
    Mish,
    FCBlock,
    Conv1DBlock,
    TransformerBlock,
)

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
        pitch = x[:,:,0]
        energy = x[:,:,1]
        pitch_reps = self.pitch_embedding(torch.bucketize(pitch, self.pitch_bins))
        energy_reps = self.energy_embedding(torch.bucketize(energy, self.energy_bins))
        prosodic_reps = pitch_reps + energy_reps
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


class VarianceAdaptor(nn.Module):
    """ Variance Adaptor """

    def __init__(self, model_config):
        super(VarianceAdaptor, self).__init__()

        self.d_model = model_config["transformer"]["encoder_hidden"]
        self.reduce_projection = nn.Linear(model_config['transformer']['encoder_hidden'] + model_config['spk_emb_dim'], model_config['transformer']['encoder_hidden'])
        if model_config['prosodic_rep_type'] == 'continuous':
            self.pros_net = ContinuousProsodicNet(model_config['prosodic_net'])
        elif model_config['prosodic_rep_type'] == 'discrete':
            self.pros_net = DiscreteProsodicNet(model_config['prosodic_net'])            
        else:    
            self.pros_net = None  

    def forward(self, x, spk_emb, pros_rep, mask, max_len):
        batch_size = x.size(0)
        # integrate speaker embedding
        if self.pros_net is not None:
            # integrate prosodic_rep
            processed_pros_rep = self.pros_net(pros_rep)
            x = x + processed_pros_rep
        spk_emb = F.normalize(spk_emb.squeeze(1)).unsqueeze(1)
        x = torch.cat([x,spk_emb.expand(batch_size, max_len, self.d_model )], dim = -1)
        x = self.reduce_projection(x)

        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)


        return x, mask
class PreNet(nn.Module):
    """ Encoder PreNet """

    def __init__(self, config):
        super(PreNet, self).__init__()
        d_model = config["transformer"]["encoder_hidden"]
        kernel_size = config["prenet"]["conv_kernel_size"]
        input_dim = config['prenet']['input_dim']
        dropout = config["prenet"]["dropout"]

        self.prenet_layer = nn.Sequential(
            Conv1DBlock(
                input_dim, d_model, kernel_size, activation=Mish(), dropout=dropout
            ),
            #Conv1DBlock(
            #    d_model, d_model, kernel_size, activation=Mish(), dropout=dropout
            #),
            #FCBlock(d_model, d_model, dropout=dropout),
        )

    def forward(self, x, mask=None):
        residual = x
        x = self.prenet_layer(x)
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)
        #x = residual + x
        return x

class Encoder(nn.Module):
    """ Encoder """

    def __init__(self, config):
        super(Encoder, self).__init__()

        n_position = config["max_len"] + 1
        d_word_vec = config["transformer"]["encoder_hidden"]
        n_layers = config["transformer"]["encoder_layer"]
        n_head = config["transformer"]["encoder_head"]
        d_k = d_v = (
            config["transformer"]["encoder_hidden"]
            // config["transformer"]["encoder_head"]
        )
        d_model = config["transformer"]["encoder_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["encoder_dropout"]

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        self.encoder_prenet = PreNet(config)
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )
        self.layer_stack = nn.ModuleList(
            [
                TransformerBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )
        #self.position_embedding_scale = torch.nn.Parameter(torch.tensor(1.0), requires_grad = True)

    def forward(self, x,  mask, return_attns=False):

        enc_slf_attn_list = []
        batch_size, max_len = x.shape[0], x.shape[1]

        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # -- PreNet
        x = self.encoder_prenet(x, mask)

        # -- Forward
        if not self.training and x.shape[1] > self.max_seq_len:
            enc_output = x + get_sinusoid_encoding_table(
                x.shape[1], self.d_model
            )[: x.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                x.device
            )
        else:
            max_len = min(max_len, self.max_seq_len)
            enc_output = x[:,:max_len,:] + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)

            mask = mask[:, :max_len]
            slf_attn_mask = slf_attn_mask[:, :max_len, :max_len]
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        return enc_output, mask
class PostNet(torch.nn.Module):
    """Postnet module for Spectrogram prediction network.

    This is a module of Postnet in Spectrogram prediction network,
    which described in `Natural TTS Synthesis by
    Conditioning WaveNet on Mel Spectrogram Predictions`_.
    The Postnet predicts refines the predicted
    Mel-filterbank of the decoder,
    which helps to compensate the detail structure of spectrogram.

    .. _`Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`:
       https://arxiv.org/abs/1712.05884

    """

    def __init__(
        self,
        idim,
        odim,
        n_layers=5,
        n_chans=512,
        n_filts=5,
        dropout_rate=0.5,
        use_batch_norm=True,
    ):
        """Initialize postnet module.

        Args:
            idim (int): Dimension of the inputs.
            odim (int): Dimension of the outputs.
            n_layers (int, optional): The number of layers.
            n_filts (int, optional): The number of filter size.
            n_units (int, optional): The number of filter channels.
            use_batch_norm (bool, optional): Whether to use batch normalization..
            dropout_rate (float, optional): Dropout rate..

        """
        super(PostNet, self).__init__()
        self.postnet = torch.nn.ModuleList()
        for layer in six.moves.range(n_layers - 1):
            ichans = odim if layer == 0 else n_chans
            ochans = odim if layer == n_layers - 1 else n_chans
            if use_batch_norm:
                self.postnet += [
                    torch.nn.Sequential(
                        torch.nn.Conv1d(
                            ichans,
                            ochans,
                            n_filts,
                            stride=1,
                            padding=(n_filts - 1) // 2,
                            bias=False,
                        ),
                        torch.nn.BatchNorm1d(ochans),
                        torch.nn.Tanh(),
                        torch.nn.Dropout(dropout_rate),
                    )
                ]
            else:
                self.postnet += [
                    torch.nn.Sequential(
                        torch.nn.Conv1d(
                            ichans,
                            ochans,
                            n_filts,
                            stride=1,
                            padding=(n_filts - 1) // 2,
                            bias=False,
                        ),
                        torch.nn.Tanh(),
                        torch.nn.Dropout(dropout_rate),
                    )
                ]
        ichans = n_chans if n_layers != 1 else odim
        if use_batch_norm:
            self.postnet += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        ichans,
                        odim,
                        n_filts,
                        stride=1,
                        padding=(n_filts - 1) // 2,
                        bias=False,
                    ),
                    torch.nn.BatchNorm1d(odim),
                    torch.nn.Dropout(dropout_rate),
                )
            ]
        else:
            self.postnet += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        ichans,
                        odim,
                        n_filts,
                        stride=1,
                        padding=(n_filts - 1) // 2,
                        bias=False,
                    ),
                    torch.nn.Dropout(dropout_rate),
                )
            ]

    def forward(self, xs):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of the sequences of padded input tensors (B, idim, Tmax).

        Returns:
            Tensor: Batch of padded output tensor. (B, odim, Tmax).

        """
        xs = xs.contiguous().transpose(1,2)
        for i in six.moves.range(len(self.postnet)):
            xs = self.postnet[i](xs)
        xs = xs.contiguous().transpose(1,2)    
        return xs
def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)

class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, config):
        super(Decoder, self).__init__()

        n_position = config["max_len"] + 1
        d_word_vec = config["transformer"]["decoder_hidden"]
        n_layers = config["transformer"]["decoder_layer"]
        n_head = config["transformer"]["decoder_head"]
        d_k = d_v = (
            config["transformer"]["decoder_hidden"]
            // config["transformer"]["decoder_head"]
        )
        d_model = config["transformer"]["decoder_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["decoder_dropout"]

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )
        
        
        self.layer_stack = nn.ModuleList(
            [
                TransformerBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, mask, return_attns=False):

        dec_slf_attn_list = []
        batch_size, max_len = x.shape[0], x.shape[1]
        

        # -- Forward
        if not self.training and x.shape[1] > self.max_seq_len:
            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = x +  get_sinusoid_encoding_table(
                x.shape[1], self.d_model
            )[: x.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                x.device
            )
        else:
            max_len = min(max_len, self.max_seq_len)
            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = x[:, :max_len, :] +  self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)
            mask = mask[:, :max_len]
            slf_attn_mask = slf_attn_mask[:, :, :max_len]

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
        return dec_output, mask

