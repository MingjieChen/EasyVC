
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from munch import Munch
from .utils import get_mask_from_lengths
import torch.nn.functional as F

def compute_loss(model, batch):
    mel, ling_rep, pros_rep, spk_emb, length, max_len = batch

    out_mel, postnet_mel, mel_mask = model(ling_rep, pros_rep, spk_emb, length, max_len)
    mae_loss = nn.L1Loss()
    
    mel_mask = ~mel_mask

    mel_target = mel[:, :mel_mask.shape[1], :]
    mel_mask = mel_mask[:, : mel_mask.shape[1]]

    mel_prediction = out_mel.masked_select(mel_mask.unsqueeze(-1))
    
    
    mel_target = mel_target.masked_select(mel_mask.unsqueeze(-1))
    if postnet_mel is not None:
        postnet_mel_prediction = postnet_mel.masked_select(mel_mask.unsqueeze(-1))
        mel_loss = mae_loss(mel_prediction, mel_target) 
        postnet_mel_loss = mae_loss(postnet_mel_prediction, mel_target)
        loss = mel_loss + postnet_mel_loss
        return loss, {'mel_loss': mel_loss.item(),
                        'total_loss': loss.item(),
                        'postnet_mel_loss': postnet_mel_loss.item()}
    else:    
        loss = mae_loss(mel_prediction, mel_target)    

        return loss, {'total_loss': loss.item()}


