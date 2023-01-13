import os, sys
import os.path as osp
import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from functools import reduce
from torch.optim import AdamW, Adam



class GeneratorScheduler:
    """ A simple wrapper class for learning rate scheduling """
    
    def __init__(self, config, optimizer, current_step = 0):
        self.optimizer = optimizer
        
        self.n_warmup_steps = config["scheduler"]["warm_up_step"]
        self.anneal_steps = config["scheduler"]["anneal_steps"]
        self.anneal_rate = config["scheduler"]["anneal_rate"]
        self.init_lr = config['optimizer']['init_lr']
        self.current_step = current_step# if current_step <= meta_learning_warmup else current_step - meta_learning_warmup

    def step(self):
        self._update_learning_rate()



    def _get_lr_scale(self):
        lr = np.min(
            [
                np.power(self.current_step, -0.5),
                np.power(self.n_warmup_steps, -1.5) * self.current_step,
            ]
        )
        for s in self.anneal_steps:
            if self.current_step > s:
                lr = lr * self.anneal_rate
        return lr

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """
        self.current_step += 1
        lr = self.init_lr * self._get_lr_scale()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
class MultiOptimizer:
    def __init__(self, optimizers={}, schedulers = {}):
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.keys = list(optimizers.keys())
        self.scheduler_keys = list(schedulers.keys())
        self.param_groups = reduce(lambda x,y: x+y, [v.param_groups for v in self.optimizers.values()])
    def state_dict(self):
        state_dicts = [(key, self.optimizers[key].state_dict())\
                       for key in self.keys]
        return state_dicts

    def load_state_dict(self, state_dict):
        for key, val in state_dict:
            try:
                self.optimizers[key].load_state_dict(val)
            except:
                print("Unloaded %s" % key)

    def step(self, key=None, scaler=None):
        keys = [key] if key is not None else self.keys
        _ = [self._step(key, scaler) for key in keys]

    def _step(self, key, scaler=None):
        if scaler is not None:
            scaler.step(self.optimizers[key])
            scaler.update()
        else:
            self.optimizers[key].step()

    def zero_grad(self, key=None):
        if key is not None:
            self.optimizers[key].zero_grad()
        else:
            _ = [self.optimizers[key].zero_grad() for key in self.keys]

    def scheduler(self, *args, key=None):
        if key is not None:
            self.schedulers[key].step(*args)
        else:
            _ = [self.schedulers[key].step(*args) for key in self.scheduler_keys]


def build_optimizer(params, config):
    #optim = dict([(key, AdamW(params, lr=config[key]['lr'], weight_decay = config[key]['weight_decay'], betas=config[key]['betas'], eps=1e-9))
    #               for key, params in parameters_dict.items()])
    
    optim = Adam(params, lr = config['optimizer']['init_lr'], weight_decay = config['optimizer']['weight_decay'],  betas=config['optimizer']['betas'], eps=1e-9)
    
    scheduler = GeneratorScheduler(config, optim)  # only generator need scheduler
    return optim, scheduler
