import os, sys
import os.path as osp
import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from functools import reduce
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import LambdaLR

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
               0.0, float(num_training_steps - current_step) /
               float(max(1, num_training_steps - num_warmup_steps))
               )

    return LambdaLR(optimizer, lr_lambda, last_epoch)
def build_optimizer(model, config):
    optimizer = torch.optim.AdamW(model.parameters(), lr = config['optimizer']['lr'],
                weight_decay = config['optimizer']['weight_decay'],
                betas = config['optimizer']['betas'],
                eps = 1e-9
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, **config['scheduler'])

    return optimizer, scheduler
