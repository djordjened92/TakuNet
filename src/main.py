import argparse
import os
import logging
from yaml import safe_dump

from torch.utils.tensorboard import SummaryWriter
import torch
from train import train
from test import test

from utils.utils import parse_command, set_random_seed, create_log_folder
from utils.distributed import is_main_process
import multiprocessing

from embedded.onnx_export import onnx_export

def setup(args: argparse.Namespace):
    if is_main_process():
        os.makedirs(args.main_runs_folder, exist_ok=True)

        args.run_path = create_log_folder(args.experiment_name, main_run_folder=args.main_runs_folder)

        config = vars(args)
        with open(os.path.join(args.run_path, 'config.yaml'), 'w') as f:
            safe_dump(config, f)

        logging.basicConfig(level=logging.INFO, filename=os.path.join(args.run_path, 'logs.txt'), filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info(args)
        logging.info("\n")


def sanity_check(args: argparse.Namespace) -> None:
    if args.distributed:
        assert torch.cuda.is_available(), "CUDA not available"
        assert "RANK" in os.environ, "RANK not set"
        assert "LOCAL_RANK" in os.environ, "LOCAL_RANK not set"
        assert "WORLD_SIZE" in os.environ, "WORLD_SIZE not set"

    assert args.num_epochs > 0, "Number of epochs must be greater than 0"
    assert args.num_workers >= 0, "Number of workers must be greater or equal to 0"
    assert args.batch_size > 0, "Batch size must be greater than 0"

    assert os.path.isabs(args.main_runs_folder), "Main runs folder must be an absolute path"
    assert os.path.isabs(args.config_path) and os.path.exists(args.config_path), "Config path must be an absolute path"

    assert len(args.class_weights) == args.num_classes, "Number of class weights must match the number of classes"
    assert args.split_type.lower() in ['emergencynet', 'ours'], f"Split type must be one of ['emergencynet', 'ours'], got {args.split_type}"

    if args.dataset.lower() == "aider":
        assert args.split.lower() in ["proportional", "exact"], "Split must be one of ['proportional', 'exact'], where exact means we use the same test size as EmergenceNet"
        assert (args.split.lower() == "proportional") and args.k_folds > 0, "K-folds must be greater than 0 for proportional split"
        assert (args.split.lower() == "exact") and (args.k_folds == 0 or args.k_fold == None), "K-folds is not used for exact split"

    assert args.mode in ['train', 'test', 'export'], "Mode must be one of ['train', 'test', 'export']"
    assert isinstance(args.aug_type, str), "Augmentation type must be a string"
    assert args.aug_type.upper() in ['AIDER'], "Augmentation type must be one of ['AIDER']"


def main() -> None:
    args = parse_command()
    args.experiment_name = ''.join([args.experiment_name, "_eval"]) if args.mode.lower() == 'test' else args.experiment_name
    if not args.mode.lower() == 'export':
        setup(args)

    set_random_seed(args.seed)
    torch.autograd.set_detect_anomaly(True) 
    
    if args.num_workers > 0:
        multiprocessing.set_start_method('spawn')

    model = None
    trainer = None
    if args.mode.lower() == 'train':
        trainer, model = train(args)

    if args.mode.lower() == 'test':
        test(trainer, model, args)

    if args.mode.lower() == 'export':
        onnx_export(args)

if __name__ == '__main__':
    main()