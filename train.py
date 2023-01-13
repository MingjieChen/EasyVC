import argparse
import os
from munch import Munch
import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset import get_dataloader
from model  import build_model
from decoder.fastspeech2.trainer import Trainer as FS2Trainer
from decoder.taco_ar.trainer import Trainer as TacoARTrainer
import random
import numpy as np
import os.path as osp
import shutil
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def main(args, config):
    
    # create exp dir
    log_dir = config['log_dir']
    exp_dir = osp.join(log_dir, config['model_name'], config['exp_name'])
    if not osp.exists(exp_dir): os.makedirs(exp_dir, exist_ok=True)
    # back up config yaml to exp dir
    if not osp.exists(osp.join(exp_dir, osp.basename(args.model_config))):
        shutil.copy(args.model_config, osp.join(exp_dir, osp.basename(args.model_config)))
    writer = SummaryWriter(exp_dir + "/tb")
    step_writer = SummaryWriter(exp_dir + '/tb')

    # dataset 
    train_loader, dev_loader = get_dataloader(config)

    # model
    model = build_model(config)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    model.to(device)

    # trainer
    trainer_class = config['trainer']
    trainer = eval(trainer_class)(  
                        args = Munch(config['loss']),
                        config = config,
                        model = model,
                        model_ema = None,
                        device = device,
                        train_dataloader = train_loader,
                        dev_dataloader = dev_loader,
                        fp16_run = config['fp16_run'],
                        step_writer = step_writer
                )
    
    if config.get('pretrained_model', '') != '':
        trainer.load_checkpoint(config['pretrained_model'],
                                load_only_params=config.get('load_only_params', True))

    # start training
    
    epochs = config['epochs']
    start_epoch = trainer.epochs
    for _ in range(start_epoch+1, epochs+1):
        epoch = trainer.epochs+1
        print(f'epoch {epoch}')
        train_results = trainer._train_epoch()
        eval_results = trainer._eval_epoch()
        results = train_results.copy()
        results.update(eval_results)
        for key, value in results.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
                writer.add_scalar(key, value, epoch)
            else:
                for v in value:
                    writer.add_figure('eval_spec', v, epoch)
        if (epoch % config['save_freq']) == 0:
            trainer.save_checkpoint(osp.join(exp_dir, f'epoch_{epoch}.pth'))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    args = parser.parse_args()

    # Read Config
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    print(model_config)
    set_seed(model_config['seed'])
    main(args, model_config)
