
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from munch import Munch
from utils import get_mask_from_lengths
import torch.nn.functional as F

def compute_loss(model, batch):
    mel, vq, speaker, length, max_len = batch

    out_mel, mel_mask = model.generator(vq, speaker, length, max_len)
    mae_loss = nn.L1Loss()
    
    mel_mask = ~mel_mask

    mel_target = mel[:, :mel_mask.shape[1], :]
    mel_mask = mel_mask[:, : mel_mask.shape[1]]

    mel_prediction = out_mel.masked_select(mel_mask.unsqueeze(-1))
    
    mel_target = mel_target.masked_select(mel_mask.unsqueeze(-1))
    loss = mae_loss(mel_prediction, mel_target)    

    return loss, {'mae': loss.item()}


def compute_generator_loss( args, model, batch):
    mel, vq, spk_emb, spk_id, length, max_len, trg_spk_emb, trg_spk_id = batch
    
    args = Munch(args)
    #out_mel, _, mel_mask = model.generator(vq, trg_spk_emb, length, max_len)
    out_mel, mel_mask = model.generator(vq, spk_emb, length, max_len)
    
    # content discriminator loss
    mse_loss = nn.MSELoss()
    #content_d_out = model.content_discriminator(vq, out_mel, max_len, mel_mask)
    #content_d_loss = mse_loss(content_d_out, torch.ones_like(content_d_out, requires_grad = False))
    # speaker_discrinator loss
    #speaker_d_out = model.speaker_discriminator(out_mel, trg_spk_id, mel_mask, max_len)
    speaker_d_out = model.speaker_discriminator(out_mel, spk_id, mel_mask, max_len)
    speaker_d_loss = mse_loss(speaker_d_out, torch.ones_like(speaker_d_out, requires_grad = False))
    
    # reconstruction loss
    #out_mel, _, mel_mask = model.generator(vq, spk_emb, length, max_len)
    mae_loss = nn.L1Loss()
    
    mel_mask = ~mel_mask

    mel_target = mel[:, :mel_mask.shape[1], :]
    mel_mask = mel_mask[:, : mel_mask.shape[1]]

    mel_prediction = out_mel.masked_select(mel_mask.unsqueeze(-1))
    mel_target = mel_target.masked_select(mel_mask.unsqueeze(-1))



    recon_loss = mae_loss(mel_prediction, mel_target) 
    #loss = args.recon * recon_loss + args.spk * speaker_d_loss + args.cont * content_d_loss
    loss = args.recon * recon_loss + args.spk * speaker_d_loss 

    return loss, {'recon': recon_loss.item(),
                   'spk_adv':speaker_d_loss.item() ,
                    }
def compute_discriminator_loss(args, model, batch):
    args = Munch(args)
    mel, vq, spk_emb, spk_id, length, max_len, trg_spk_emb, trg_spk_id = batch
    #mel_mask = get_mask_from_lengths(us_length, us_max_len)
                        
    mse_loss = nn.MSELoss()
    #out_mel, _, mel_mask = model.generator(vq, trg_spk_emb, length, max_len)
    out_mel, mel_mask = model.generator(vq, spk_emb, length, max_len)
    #fake_content_d_out = model.content_discriminator(vq, out_mel, max_len, mel_mask)
    #fake_content_d_loss = mse_loss(fake_content_d_out, torch.zeros_like(fake_content_d_out, requires_grad = False))
    
    #fake_speaker_d_out = model.speaker_discriminator(out_mel, trg_spk_id, mel_mask, max_len)
    fake_speaker_d_out = model.speaker_discriminator(out_mel, spk_id, mel_mask, max_len)
    fake_speaker_d_loss = mse_loss(fake_speaker_d_out, torch.zeros_like(fake_speaker_d_out, requires_grad = False))
    
    #real_content_d_out = model.content_discriminator(vq, mel, max_len, mel_mask)
    #real_content_d_loss = mse_loss(real_content_d_out, torch.ones_like(real_content_d_out, requires_grad = False))
    
    real_speaker_d_out = model.speaker_discriminator(mel, spk_id, mel_mask, max_len)
    real_speaker_d_loss = mse_loss(real_speaker_d_out, torch.ones_like(real_speaker_d_out, requires_grad = False))
    
    # prototype loss
    prototype = model.speaker_discriminator.speaker_embs.weight.contiguous().transpose(0,1)

    #spk_logits = torch.matmul(trg_spk_emb.squeeze(1), prototype)
    spk_logits = torch.matmul(spk_emb.squeeze(1), prototype)
    #spk_loss = F.cross_entropy(spk_logits, trg_spk_id.squeeze(1).long())
    spk_loss = F.cross_entropy(spk_logits, spk_id.squeeze(1).long())
    #loss = real_speaker_d_loss + real_content_d_loss + fake_speaker_d_loss + fake_content_d_loss + spk_loss   
    loss = real_speaker_d_loss + fake_speaker_d_loss + spk_loss
    return loss, {
                    'spk_real':real_speaker_d_loss.item(),
                    'spk_fake':fake_speaker_d_loss.item(),
                    "spk_loss": spk_loss.item()
                    }
"""
def compute_generator_loss( args, model, batch):
    mel, vq, speaker, length, max_len, \
        us_mel, us_speaker_emb, us_speaker_id, us_trg_speaker_emb, us_trg_speaker_id, us_length, us_max_len = batch
    
    args = Munch(args)
    out_mel, _, mel_mask = model.generator(vq, us_trg_speaker_emb, length, max_len)
    
    # content discriminator loss
    mse_loss = nn.MSELoss()
    content_d_out = model.content_discriminator(vq, out_mel, max_len, mel_mask)
    content_d_loss = mse_loss(content_d_out, torch.ones_like(content_d_out, requires_grad = False))
    # speaker_discrinator loss
    speaker_d_out = model.speaker_discriminator(out_mel, us_trg_speaker_emb, mel_mask, max_len)
    speaker_d_loss = mse_loss(speaker_d_out, torch.ones_like(speaker_d_out, requires_grad = False))
    
    # reconstruction loss
    out_mel, _, mel_mask = model.generator(vq, speaker, length, max_len)
    mae_loss = nn.L1Loss()
    
    mel_mask = ~mel_mask

    mel_target = mel[:, :mel_mask.shape[1], :]
    mel_mask = mel_mask[:, : mel_mask.shape[1]]

    mel_prediction = out_mel.masked_select(mel_mask.unsqueeze(-1))
    mel_target = mel_target.masked_select(mel_mask.unsqueeze(-1))



    recon_loss = mae_loss(mel_prediction, mel_target) 
    loss = recon_loss + args.spk * speaker_d_loss + args.cont * content_d_loss

    return loss, {'recon': recon_loss.item(),
                   'spk_adv':speaker_d_loss.item() ,
                   'cont_adv': content_d_loss.item()
                    }
def compute_discriminator_loss(args, model, batch):
    args = Munch(args)
    mel, vq, speaker, length, max_len, \
        us_mel, us_speaker_emb, us_speaker_id, us_trg_speaker_emb, us_trg_speaker_id, us_length, us_max_len = batch
    mel_mask = get_mask_from_lengths(us_length, us_max_len)
                        
    mse_loss = nn.MSELoss()
    out_mel, _, us_mel_mask = model.generator(vq, us_trg_speaker_emb, length, max_len)
    fake_content_d_out = model.content_discriminator(vq, out_mel, max_len, us_mel_mask)
    fake_content_d_loss = mse_loss(fake_content_d_out, torch.zeros_like(fake_content_d_out, requires_grad = False))
    
    fake_speaker_d_out = model.speaker_discriminator(out_mel, us_trg_speaker_emb, us_mel_mask, max_len)
    fake_speaker_d_loss = mse_loss(fake_speaker_d_out, torch.zeros_like(fake_speaker_d_out, requires_grad = False))
    
    real_content_d_out = model.content_discriminator(vq, mel, max_len, us_mel_mask)
    real_content_d_loss = mse_loss(real_content_d_out, torch.ones_like(real_content_d_out, requires_grad = False))
    
    real_speaker_d_out = model.speaker_discriminator(us_mel, us_speaker_emb, mel_mask, us_max_len)
    real_speaker_d_loss = mse_loss(real_speaker_d_out, torch.ones_like(real_speaker_d_out, requires_grad = False))
    
    loss = real_speaker_d_loss + real_content_d_loss + fake_speaker_d_loss + fake_content_d_loss 
    #loss = real_speaker_d_loss + fake_speaker_d_loss
    return loss, {
                    'spk_real':real_speaker_d_loss.item(),
                    'spk_fake':fake_speaker_d_loss.item(),
                    'cont_real':real_content_d_loss.item(),
                    'cont_fake': fake_content_d_loss.item()
                    }
"""                    
