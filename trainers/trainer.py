
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
from loss import compute_loss



class Trainer(object):
    def __init__(self,
                 args = None,
                 model=None,
                 model_ema=None,
                 optimizer=None,
                 scheduler=None,
                 config={},
                 device=torch.device("cpu"),
                 train_dataloader=None,
                 dev_dataloader=None,
                 initial_steps=0,
                 initial_epochs=0,
                 fp16_run=False,
                 step_writer = None
    ):
        
        self.args = args
        self.steps = initial_steps
        self.epochs = initial_epochs
        self.model = model
        self.model_ema = model_ema
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.config = config
        self.device = device
        self.finish_train = False
        self.fp16_run = fp16_run
        self.step_writer = step_writer
        print(f'trainer device {self.device}')
        self.iters = 0

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.
        Args:
            checkpoint_path (str): Checkpoint path to be saved.
        """
        state_dict = {
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
            "epochs": self.epochs,
            "model": {key: self.model[key].state_dict() for key in self.model},
            "iters": self.iters
        }
        if self.model_ema is not None:
            state_dict['model_ema'] = {key: self.model_ema[key].state_dict() for key in self.model_ema}

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
        for key in self.model:
            self._load(state_dict["model"][key], self.model[key])

        if self.model_ema is not None:
            for key in self.model_ema:
                self._load(state_dict["model_ema"][key], self.model_ema[key])
        
        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.iters = state_dict['iters']
            self.optimizer.load_state_dict(state_dict["optimizer"])
            self.optimizer.schedulers['generator'].current_step = self.iters


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
        for param_group in self.optimizer.optimizers['generator'].param_groups:
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
        _ = [self.model[k].train() for k in self.model]
        scaler = torch.cuda.amp.GradScaler() if (('cuda' in str(self.device)) and self.fp16_run) else None

        
        for train_steps_per_epoch, batchs in tqdm(enumerate(self.train_dataloader, 1)):
            for batch in batchs:
                #batch = [b.to(self.device) for b in batch ]
                _batch = []
                shapes = ""
                for b in batch:
                    if isinstance(b, torch.Tensor):
                        _batch.append(b.to(self.device))
                        #shapes += f' {b.size()}  '
                    else:
                        _batch.append(b)    
                
                #print(f'shapes {shapes}', flush = True)        
                
                self.optimizer.zero_grad()
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        loss, losses = compute_loss(self.model, _batch)
                    scaler.scale(loss).backward()
                else:
                    loss, losses = compute_loss(self.model, _batch)        
                    loss.backward()
                self.optimizer.step('generator', scaler=scaler)
                loss_string = f" epoch {self.epochs}, iters {self.iters}" 
                for key in losses:
                    train_losses["train/%s" % key].append(losses[key])
                    loss_string += f" {key}:{losses[key]:.5f} "
                    self.step_writer.add_scalar('step/'+key, losses[key], self.iters)
                self.step_writer.add_scalar('step/lr', self._get_lr(), self.iters)    
                #print(loss_string, flush = True)    
                #print()
                self.iters+=1
                self.optimizer.scheduler()
        train_losses = {key: np.mean(value) for key, value in train_losses.items()}
        return train_losses
    
    
    @torch.no_grad()
    def _eval_epoch(self):
        
        eval_losses = defaultdict(list)
        _ = [self.model[k].eval() for k in self.model]
        for eval_steps_per_epoch, batchs in enumerate(self.dev_dataloader, 1):
            for batch in batchs:
                _batch = []
                for b in batch:
                    if isinstance(b, torch.Tensor):
                        _batch.append(b.to(self.device))
                    else:
                        _batch.append(b)    
                loss, losses = compute_loss(self.model, _batch)        
                for key in losses:
                    eval_losses["eval/%s" % key].append(losses[key])
        
        eval_losses = {key: np.mean(value) for key, value in eval_losses.items()}
        eval_string = f"epoch {self.epochs}, eval: "
        for key, value in eval_losses.items():
            eval_string += f"{key}: {value:.6f} "
        #print(eval_string, flush = True)    
        #print()
        return eval_losses