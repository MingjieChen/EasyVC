
import os
import os.path as osp
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from munch import Munch
import torch.nn.functional as F
from .losses import compute_g_loss, compute_d_loss
from .commons import  clip_grad_value_

class Trainer(object):
    def __init__(self,
                 args = None,
                 model=None,
                 model_ema=None,
                 config={},
                 device=torch.device("cpu"),
                 train_dataloader=None,
                 dev_dataloader=None,
                 initial_steps=0,
                 initial_epochs=0,
                 fp16_run=False,
                 step_writer = None,
                 timer = None
    ):
        
        self.args = args
        self.epochs = initial_epochs
        self.model_ema = model_ema
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.config = config
        self.device = device
        self.finish_train = False
        self.fp16_run = fp16_run
        self.step_writer = step_writer
        self.timer = timer
        if self.config['ngpu'] > 1:
            self.model = model.module
        else:    
            self.model = model
        print(f'trainer device {self.device}')
        self.iters = 0
        self.optim_g = torch.optim.AdamW(
          self.model.generator.parameters(), 
          config['optimizer']['generator']['lr'], 
          config['optimizer']['generator']['betas'], 
          eps=config['optimizer']['generator']['eps'])
        self.optim_d = torch.optim.AdamW(
          self.model.discriminator.parameters(),
          config['optimizer']['discriminator']['lr'], 
          config['optimizer']['discriminator']['betas'], 
          eps=config['optimizer']['discriminator']['eps'])
            
        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(self.optim_g, gamma=config['scheduler']['generator']['lr_decay'], last_epoch=self.epochs - 1)
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.optim_d, gamma=config['scheduler']['discriminator']['lr_decay'], last_epoch=self.epochs - 1)

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.
        Args:
            checkpoint_path (str): Checkpoint path to be saved.
        """
        state_dict = {
            "optim_g": self.optim_g.state_dict(),
            "optim_d": self.optim_d.state_dict(),
            "sched_g": self.scheduler_g.state_dict(),
            "sched_d": self.scheduler_d.state_dict(),
            "epochs": self.epochs,
            "model": self.model.state_dict(),
            "iters": self.iters
        }
        if self.model_ema is not None:
            state_dict['model_ema'] = self.model_ema.state_dict()

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.

        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self._load(state_dict["model"], self.model)

        if self.model_ema is not None:
            self._load(state_dict["model_ema"], self.model_ema)
        
        if not load_only_params:
            self.epochs = state_dict["epochs"]
            self.iters = state_dict['iters']
            self.optim_g.load_state_dict(state_dict["optim_g"])
            self.optim_d.load_state_dict(state_dict["optim_d"])
            self.scheduler_g.load_state_dict(state_dict['sched_g'])
            self.scheduler_d.load_state_dict(state_dict['sched_d'])


    def _load(self, states, model, force_load=True):
        model_states = model.state_dict()
        for key, val in states.items():
            try:
                if key not in model_states:
                    continue
                if isinstance(val, nn.Parameter):
                    val = val.data

                if val.shape != model_states[key].shape:
                    self.logger.info("%s does not have same shape" % key)
                    print(val.shape, model_states[key].shape)
                    if not force_load:
                        continue

                    min_shape = np.minimum(np.array(val.shape), np.array(model_states[key].shape))
                    slices = [slice(0, min_index) for min_index in min_shape]
                    model_states[key][slices].copy_(val[slices])
                else:
                    model_states[key].copy_(val)
            except:
                print("not exist ", key)

    @staticmethod
    def get_gradient_norm(model):
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

        total_norm = np.sqrt(total_norm)
        return total_norm
    def _get_lr(self):
        for param_group in self.optim_g.param_groups:
            lr = param_group['lr']
            break
        return lr

    @staticmethod
    def moving_average(model, model_test, beta=0.999):
        for param, param_test in zip(model.parameters(), model_test.parameters()):
            param_test.data = torch.lerp(param.data, param_test.data, beta)
    
    
    def _train_epoch(self):
        self.epochs += 1
        
        train_losses = defaultdict(list)
        self.model.train()
        scaler = torch.cuda.amp.GradScaler() if (('cuda' in str(self.device)) and self.fp16_run) else None

        
        for train_steps_per_epoch, batch in tqdm(enumerate(self.train_dataloader, 1), total = len(self.train_dataloader)):
            _batch = []
            for b in batch:
                if isinstance(b, torch.Tensor):
                    _batch.append(b.to(self.device))
                else:
                    _batch.append(b)    
            self.optim_g.zero_grad()
            self.optim_d.zero_grad()
            if scaler is not None:
                d_loss, d_losses = compute_d_loss(self.model, _batch, self.config)
                scaler.scale(d_loss).backward()
                scaler.unscale_(self.optim_d)
                grad_norm_d = clip_grad_value_(self.model.discriminator.parameters(), None)
                scaler.step(self.optim_d)
                
                
                g_loss, g_losses = compute_g_loss(self.model, _batch, self.config)
                scaler.scale(g_loss).backward()
                scaler.unscale_(self.optim_g)
                grad_norm_g = clip_grad_value_(self.model.generator.parameters(), None)
                scaler.step(self.optim_g)

                scaler.update()
            else:
                d_loss, d_losses = compute_d_loss(self.model, _batch, self.config)
                d_loss.backward()
                self.optim_d.step()
                grad_norm_d = clip_grad_value_(self.model.discriminator.parameters(), None)
                
                g_loss, g_losses = compute_g_loss(self.model, _batch, self.config)
                g_loss.backward()
                grad_norm_g = clip_grad_value_(self.model.generator.parameters(), None)
                self.optim_g.step()
            
            loss_string = f"epoch: {self.epochs}| iters: {self.iters}|" 
            for key in d_losses:
                train_losses["train/%s" % key].append(d_losses[key])
                loss_string += f" {key}:{d_losses[key]:.3f} "
                self.step_writer.add_scalar('step/'+key, d_losses[key], self.iters)
            for key in g_losses:
                train_losses["train/%s" % key].append(g_losses[key])
                loss_string += f" {key}:{g_losses[key]:.3f} "
                self.step_writer.add_scalar('step/'+key, g_losses[key], self.iters)
            self.step_writer.add_scalar('step/lr', self._get_lr(), self.iters)    
            self.iters+=1
            if self.iters % self.config['show_freq'] == 0:
                print(loss_string, flush = True)
        self.scheduler_g.step()
        self.scheduler_d.step()
        train_losses = {key: np.mean(value) for key, value in train_losses.items()}
        return train_losses
    
    
    @torch.no_grad()
    def _eval_epoch(self):
        
        eval_losses = defaultdict(list)
        self.model.eval()
        for eval_steps_per_epoch, batch in tqdm(enumerate(self.dev_dataloader, 1), total = len(self.dev_dataloader)):
            _batch = []
            for b in batch:
                if isinstance(b, torch.Tensor):
                    _batch.append(b.to(self.device))
                else:
                    _batch.append(b)    
            d_loss, d_losses = compute_d_loss(self.model, _batch, self.config)        
            g_loss, g_losses = compute_g_loss(self.model, _batch, self.config)        
            for key in d_losses:
                eval_losses["eval/%s" % key].append(d_losses[key])
            for key in g_losses:
                eval_losses["eval/%s" % key].append(g_losses[key])
        
        eval_losses = {key: np.mean(value) for key, value in eval_losses.items()}
        eval_string = f"epoch {self.epochs}, eval: "
        for key, value in eval_losses.items():
            eval_string += f"{key}: {value:.6f} "
        
        return eval_losses
