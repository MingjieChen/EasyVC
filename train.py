import argparse
import os
from munch import Munch
import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset import get_dataloader
from model  import build_model
from decoder.fastspeech2.trainer import Trainer as FS2Trainer
from decoder.taco_ar.trainer import Trainer as TacoARTrainer
from decoder.taco_mol.trainer import Trainer as TacoMOLTrainer
from decoder.vits.trainer import Trainer as VITSTrainer
from decoder.grad_tts.trainer import Trainer as GradTTSTrainer
import random
import numpy as np
import os.path as osp
import shutil
import torch.multiprocessing as mp
import torch.distributed as dist
from util import Timer
def clean_checkpoints(path_to_models='ckpt', n_ckpts_to_keep=10, sort_by_time=True, prefix = 'epoch'):
    ckpts_files = [f for f in os.listdir(path_to_models) if os.path.isfile(os.path.join(path_to_models, f))]
    name_key = (lambda _f: int(re.compile('._(\d+)\.pth').match(_f).group(1)))
    time_key = (lambda _f: os.path.getmtime(os.path.join(path_to_models, _f)))
    sort_key = time_key if sort_by_time else name_key
    x_sorted = lambda _x: sorted([f for f in ckpts_files if f.startswith(_x) and not f.endswith('_0.pth')], key=sort_key)
    to_del = [os.path.join(path_to_models, fn) for fn in
             x_sorted(prefix)[:-n_ckpts_to_keep]]
    del_info = lambda fn: print(f".. Free up space by deleting ckpt {fn}")
    del_routine = lambda x: [os.remove(x), del_info(x)]
    rs = [del_routine(fn) for fn in to_del]
def _get_free_port():
    import socketserver
    with socketserver.TCPServer(('localhost',0),None) as s:
        return s.server_address[1]
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def main(replica_id = None, replica_count = None, port = None, args = None, config = None):
    
    set_seed(config['seed'])
    # ddp set up
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    if replica_id is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    else:
        dist.init_process_group(backend = 'nccl', world_size = replica_count, rank = replica_id)
        device = torch.device("cuda", replica_id )
    
        torch.cuda.set_device(device)

    # create exp dir
    log_dir = args.log_dir
    exp_dir = osp.join(log_dir, args.model_name, args.exp_name)
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
    
    model.to(device)

    if replica_id is not None:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[replica_id])
        cudnn.benchmark = True

    # trainer
    timer = Timer()
    timer.set()
    trainer_class = config['trainer']
    trainer = eval(trainer_class)(  
                        args = Munch(config['losses']),
                        config = config,
                        model = model,
                        model_ema = None,
                        device = device,
                        train_dataloader = train_loader,
                        dev_dataloader = dev_loader,
                        fp16_run = config['fp16_run'],
                        step_writer = step_writer,
                        timer = timer
                )
    
    if args.pretrained_model != '':
        trainer.load_checkpoint(args.pretrained_model,
                                load_only_params=config.get('load_only_params', True))

    # start training
    
    max_epochs = config['epochs']
    start_epoch = trainer.epochs
    for _ in range(start_epoch+1, max_epochs+1):
        epoch = trainer.epochs+1
        print(f'start epoch {epoch}')
        train_results = trainer._train_epoch()
        eval_results = trainer._eval_epoch()
        results = train_results.copy()
        results.update(eval_results)
        loss_string = f"epoch {epoch} |"
        for key, value in results.items():
            if isinstance(value, float):
                loss_string += f" {key}: {value:.4f} "
                writer.add_scalar(key, value, epoch)
            else:
                for v in value:
                    writer.add_figure('eval_spec', v, epoch)
        
        
        # save            
        if epoch % config['save_freq'] == 0:
            if replica_id == 0 or replica_id == None:
                trainer.save_checkpoint(osp.join(exp_dir, 'ckpt',f'epoch_{epoch}.pth'))
                clean_checkpoints(osp.join(exp_dir, 'ckpt'), prefix = 'epoch' )




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument("-p","--pretrained_model", type = str, default = "", help = "model checkpoint to be resumed for training")
    parser.add_argument("-e", "--exp_name", type = str, default = "", help="experiment name")
    parser.add_argument("-l", "--log_dir", type = str, default = "exp", help="experiment root dir")
    parser.add_argument("-m", "--model_name", type = str, default = "", help="model name, e.g. vqw2v_uttdev_f0_fs2")
    args = parser.parse_args()

    # Read Config
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    print(model_config)
    print(args)
    if model_config['ngpu'] > 1:
        replica_count = torch.cuda.device_count()
        print(f'Using {replica_count} GPUs')
        if replica_count >1:
            port = _get_free_port()
            mp.spawn(main, args=(replica_count, port, args, model_config), nprocs = replica_count, join = True)
    else:
        main(args = args, config = model_config)
