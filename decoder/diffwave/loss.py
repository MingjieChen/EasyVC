import torch
import numpy as np
import torch.nn as nn

loss_fn = nn.L1Loss()
noise_schedule = np.linspace(1e-4, 0.05, 50).tolist()
def compute_loss(model, batch):
    audio, _, ling_rep, pros_rep, spk_emb, lengths, _ = batch 
    
    audio = audio.squeeze(1)
    N, T = audio.shape
    
    beta = np.array(noise_schedule)
    noise_level = np.cumprod(1 - beta)
    noise_level = torch.tensor(noise_level.astype(np.float32))
    
    noise_level = noise_level.to(audio.device)
    
    t = torch.randint(0, len(noise_schedule), [N], device=audio.device)
    noise_scale = noise_level[t].unsqueeze(1)
    noise_scale_sqrt = noise_scale**0.5
    noise = torch.randn_like(audio)
    noisy_audio = noise_scale_sqrt * audio + (1.0 - noise_scale)**0.5 * noise
    predicted = model(noisy_audio, t, ling_rep, pros_rep, spk_emb, lengths)
    loss = loss_fn(noise, predicted.squeeze(1))
    
    losses = {'diff_loss': loss.item()}
    
    return loss, losses 
