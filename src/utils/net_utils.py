import argparse
from torch import nn

from networks.Builder import create_takunet

def extract_net_params(args: argparse.Namespace):
    return {
        'network': args.network,
        'dataset': args.dataset,
        'input_channels': args.input_channels,
        'output_classes': args.num_classes,
        'dense': args.dense,
        'stem_reduction': args.stem_reduction,
        'k_folds': args.k_folds,
        'split': args.split,
        'resolution': args.img_width,
    }

def extract_optim_params(args: argparse.Namespace):
    return {
        'optimizer': args.optimizer,
        'scheduler': args.scheduler,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'scheduler_per_epoch': args.scheduler_per_epoch,
        'learning_rate': args.learning_rate,
        'learning_rate_decay': args.learning_rate_decay,
        'learning_rate_decay_steps': args.learning_rate_decay_steps,
        'min_learning_rate': args.min_learning_rate,
        'warmup_epochs': args.warmup_epochs,
        'warmup_steps': args.warmup_steps,
        'weight_decay': args.weight_decay,
        'weight_decay_end': args.weight_decay_end,
        'update_freq': args.update_freq,
        'alpha': args.alpha,
        'momentum': args.momentum,
        'model_ema': args.model_ema,
    }

def select_arch(net_kwargs: dict, criterion: nn.Module, optim_kwargs: dict, ckpt_path: str = None):
    if net_kwargs['network'].lower() == 'takunet':
        return create_takunet(net_kwargs, criterion, optim_kwargs, ckpt_path)
    else:
        raise ValueError(f"Unsupported network: {net_kwargs['network']}")
