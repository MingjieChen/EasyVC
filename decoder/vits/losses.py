import torch 
from torch.nn import functional as F
from .mel_processing import mel_spectrogram_torch, spec_to_mel_torch

from torch.cuda.amp import autocast
from .commons import slice_segments
def compute_g_loss(model, batch, config):
    y, spec, ling, pros, spk_emb, spec_lengths, audio_length = batch 
    with autocast(enabled=config['fp16_run']):
        y_hat, ids_slice, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = model.generator(spec, spec_lengths, ling, spk_emb, pros)

        mel = spec_to_mel_torch(
           spec,
           config['decoder_params']['filter_length'],
           config['decoder_params']['n_mels_channels'],
           config['decoder_params']['sampling_rate'],
        )
    y_mel = slice_segments(mel, ids_slice, config['decoder_params']['segment_size'] // config['decoder_params']['hop_length'])
    y_hat = y_hat.float()
    y_hat_mel = mel_spectrogram_torch(
          y_hat.squeeze(1), 
          config['decoder_params']['filter_length'], 
          config['decoder_params']['n_mels_channels'], 
          config['decoder_params']['sampling_rate'], 
          config['decoder_params']['hop_length'], 
          config['decoder_params']['win_length'], 
      )
    y = slice_segments(y, ids_slice * config['decoder_params']['hop_length'], config['decoder_params']['segment_size']) # slice 
    assert y.size(2) == y_hat.size(2), f'y {y.size()} y_hat {y_hat.size()}'

    with autocast(enabled=config['fp16_run']):
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = model.discriminator(y, y_hat)
    with autocast(enabled=False):
        loss_mel = F.l1_loss(y_mel, y_hat_mel) * config['losses']['mel']
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, spec_mask) * config['losses']['kl']

        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, losses_gen = generator_loss(y_d_hat_g)
        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
    return loss_gen_all, {'loss_gen_all': loss_gen_all.item(),
                           'loss_gen': loss_gen.item(),
                           'loss_fm': loss_fm.item(),
                           'loss_mel': loss_mel.item(),
                           'loss_kl': loss_kl.item() 
                            }
def compute_d_loss(model, batch, config):
    y, spec, ling, pros, spk_emb, spec_lengths, audio_length = batch 

    with autocast(enabled=config['fp16_run']):
        y_hat, ids_slice, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = model.generator(spec, spec_lengths, ling, spk_emb, pros)
        y_d_hat_r, y_d_hat_g, _, _ = model.discriminator(y, y_hat.detach())
    with autocast(enabled=False):
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
        loss_disc_all = loss_disc
    return loss_disc_all, {'loss_disc_all': loss_disc_all.item(),
                            'loss_disc_r': losses_disc_r.item(),
                            'loss_disc_g': losses_disc_g.item()}    
    
def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2 


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = 0
    g_losses = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        #r_losses.append(r_loss.item())
        #g_losses.append(g_loss.item())
        r_losses += r_loss
        g_losses += g_loss

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.float()
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
            z_p, logs_q: [b, h, t_t]
            m_p, logs_p: [b, h, t_t]
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p)
    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)
    return l
