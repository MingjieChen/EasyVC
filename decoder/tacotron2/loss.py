import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from utils import make_non_pad_mask

class Loss(nn.Module):
    """
    L1 loss module supporting (1) loss calculation in the normalized target feature space
                              (2) masked loss calculation
    """
    def __init__(self):
        super(Loss, self).__init__()
        self.objective = torch.nn.L1Loss(reduction="mean")


    def forward(self, x, y, x_lens, y_lens, device):
        # match the input feature length to acoustic feature length to calculate the loss
        if x.shape[1] > y.shape[1]:
            x = x[:, :y.shape[1]]
            masks = make_non_pad_mask(y_lens).unsqueeze(-1).to(device)
        if x.shape[1] <= y.shape[1]:
            y = y[:, :x.shape[1]]
            masks = make_non_pad_mask(x_lens).unsqueeze(-1).to(device)
        
        # calculate masked loss
        x_normalized = x #self.normalize(x)
        y_normalized = y #self.normalize(y.to(device))
        x_masked = x_normalized.masked_select(masks)
        y_masked = y_normalized.masked_select(masks)
        loss = self.objective(x_masked, y_masked)
        return loss 

def compute_loss(model, batch, objective, *args, **kwargs ) :
    
    mel, vq, speaker, length, max_len = batch
    device = mel.device
    predicted_features, predicted_feature_lengths = model(vq, length, speaker, mel)
    # loss calculation (masking and normalization are done inside)
    loss = objective(predicted_features,
                        mel,
                        predicted_feature_lengths,
                        length,
                        device)
    return loss, {'mae':loss.item()}
           
