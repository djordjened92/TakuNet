import argparse
import os
import datetime
from yaml import safe_load


import random
import numpy as np
import torch
try:
    from lightning.pytorch import seed_everything
except:
    pass

def parse_command():
    parser = argparse.ArgumentParser(description='Command Parser')
    
    # Basics
    parser.add_argument('--config-path', type=str, help='path to the configuration file')
    parser.add_argument('--experiment-name', type=str, help='name of the experiment')
    parser.add_argument('--num-epochs', type=int, help='train number of epochs')
    parser.add_argument('--seed', type=int, default=22, help='random seed')
    parser.add_argument('--main-runs-folder', type=str, default='runs', help='main folder to store runs')
    parser.add_argument('--mode', type=str, default='train', help='mode to run the script')
    parser.add_argument('--lightning-precision', type=str, default='mixed', help='precision used for lightning')

    # Loggers
    parser.add_argument('--wandb', action='store_true', default=False, help='whether to use wandb')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='whether to use tensorboard')

    # Dataloader
    parser.add_argument('--dataset', type=str, default="AIDER", help='name of the dataset')
    parser.add_argument('--data-path', type=str, default='./data', help='path to the data')
    parser.add_argument('--num-workers', type=int, default=4, help='number of workers used for dataloading')
    parser.add_argument('--persistent-workers', action='store_true', default=False, help='whether to use persistent workers')
    parser.add_argument('--img-height', type=int, default=240, help='image height')
    parser.add_argument('--img-width', type=int, default=240, help='image width')
    parser.add_argument('--num-classes', type=int, default=5, help='number of classes in the dataset')
    parser.add_argument('--pin-memory', action='store_true', default=False, help='whether to use pin memory')
    parser.add_argument('--augment', type=str, default=None, help='augmentation type')
    parser.add_argument('--k-folds', type=int, default=0, help='number of folds for cross validation')
    parser.add_argument('--split', type=str, default='proportional', help='split used for the dataset')
    parser.add_argument('--no-validation', action='store_true', default=False, help='whether to use validation set')

    # Network
    parser.add_argument('--network', type=str, default='takunet', help='network used for training')
    parser.add_argument('--input-channels', type=int, default=3, help='number of input channels')
    parser.add_argument('--ckpts-path', type=str, help='path to the model checkpoint')
    parser.add_argument('--dense', action='store_true', default=True, help='whether to use dense connections')
    parser.add_argument('--stem-reduction', type=int, default=4, help='reduction factor for the stem')

    # Optimization
    parser.add_argument('--optimizer', type=str, default='rmsprop', help='optimizer used for training')
    parser.add_argument('--scheduler', type=str, default='step', help='scheduler used for training')
    parser.add_argument('--batch-size', type=int, help='batch size used for training')
    parser.add_argument('--update-freq', default=1, type=int, help='gradient accumulation steps')
    parser.add_argument('--learning-rate', type=float, default=4e-3, help='learning rate used for training')
    parser.add_argument('--learning-rate-decay', type=float, default=0.1, help='learning rate decay used for training')
    parser.add_argument('--learning-rate-decay-steps', type=int, default=30, help='learning rate decay steps used for training')
    parser.add_argument('--min-learning-rate', type=float, default=1e-6, help='min learning rate used for training')
    parser.add_argument('--scheduler-per-epoch', action='store_true', default=False, help='whether to scale learning rate per epoch or per total iterations')
    parser.add_argument('--warmup-epochs', type=int, default=10, help='number of warmup epochs')
    parser.add_argument('--warmup-steps', type=int, default=-1, help='number of warmup steps')
    parser.add_argument('--weight-decay', type=float, default=0.05, help='weight decay used for training')
    parser.add_argument('--weight-decay-end', type=float, default=None, help="""Final value of the -weight decay.""")
    parser.add_argument('--opt-eps', type=float, default=1e-8, help='optimizer epsilon')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA', help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='label smoothing value')
    parser.add_argument('--model-ema', action='store_true', default=False, help='whether to use model ema')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum value')
    parser.add_argument('--alpha', type=float, default=0.9, help='alpha value for RMSprop')
    parser.add_argument('--class-weights', type=float, nargs='+', help='class weight used for training')

    # Export
    parser.add_argument('--onnx-opset-version', type=int, default=17, help='onnx opset version')
    
    
    args = parser.parse_args()
    args = load_arguments_from_config(args)
    
    return args

def load_arguments_from_config(args: argparse.Namespace):
    if os.path.exists(args.config_path):
        with open(args.config_path, 'r') as f:
            config = safe_load(f)
        
        for key, value in config.items():
            setattr(args, key, value)
    return args

def set_random_seed(seed: int=42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        seed_everything(seed)
    except:
        pass

def create_log_folder(experiment_name: str, main_run_folder: str="runs"):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    folder_name = current_time if experiment_name is None or experiment_name == "" else experiment_name
    log_folder = os.path.join(main_run_folder, folder_name)

    os.makedirs(log_folder, exist_ok=True)
    os.makedirs(os.path.join(log_folder, "images"), exist_ok=True)

    return log_folder
