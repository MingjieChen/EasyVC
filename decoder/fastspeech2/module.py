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
class VarianceAdaptor(nn.Module):
    """ Variance Adaptor """

    def __init__(self, model_config):
        super(VarianceAdaptor, self).__init__()

        self.d_model = model_config["transformer"]["encoder_hidden"]
        self.projection = nn.Linear(model_config['spk_emb_dim'],model_config['transformer']['encoder_hidden'])

    def forward(self, x, spk_emb, f0, mask, max_len):
        batch_size = x.size(0)
        spk_emb = self.projection(spk_emb.squeeze(1)).unsqueeze(1)
        x = x + spk_emb.expand(batch_size, max_len, self.d_model )
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)


        return x, mask

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

        self.encoder_prenet = EncoderPreNet(config)
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
class EncoderPreNet(nn.Module):
    """ Phoneme Encoder PreNet """

    def __init__(self, config):
        super(EncoderPreNet, self).__init__()
        d_model = config["transformer"]["encoder_hidden"]
        kernel_size = config["prenet"]["conv_kernel_size"]
        input_dim = config['prenet']['input_dim']
        dropout = config["prenet"]["dropout"]

        self.prenet_layer = nn.Sequential(
            Conv1DBlock(
                input_dim, d_model, kernel_size, activation=Mish(), dropout=dropout
            ),
            Conv1DBlock(
                d_model, d_model, kernel_size, activation=Mish(), dropout=dropout
            ),
            FCBlock(d_model, d_model, dropout=dropout),
        )

    def forward(self, x, mask=None):
        residual = x
        x = self.prenet_layer(x)
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)
        #x = residual + x
        return x
class Postnet(torch.nn.Module):
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
        super(Postnet, self).__init__()
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
        for i in six.moves.range(len(self.postnet)):
            xs = self.postnet[i](xs)
        return xs
class DecoderPreNet(nn.Module):
    """ Mel-spectrogram Decoder PreNet """

    def __init__(self, config):
        super(DecoderPreNet, self).__init__()
        input_dim = config["transformer"]["encoder_hidden"]
        d_encoder = config["transformer"]["encoder_hidden"]
        d_out = config['transformer']['decoder_hidden']
        dropout = config["prenet"]["dropout"]

        self.prenet_layer = nn.Sequential(
            FCBlock(input_dim, d_encoder, activation=Mish(), dropout=dropout),
            FCBlock(d_encoder, d_out, activation=Mish(), dropout=dropout),
        )

    def forward(self, x, mask=None):
        x = self.prenet_layer(x)
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)
        return x
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

        self.decoder_prenet = DecoderPreNet(config)
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
        

        # -- PreNet
        x = self.decoder_prenet(x)
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

