import argparse
import logging
import os
import wandb

import torch
import torch.nn as nn
from torchinfo import summary

import lightning as L
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger

from utils.distributed import get_rank, is_main_process
from utils.net_utils import extract_net_params, extract_optim_params
from fvcore.nn import FlopCountAnalysis

from utils.net_utils import select_arch
from datasets.dataloader import get_dataset, get_dataloader

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.metrics.cam_mult_image import CamMultImageConfidenceChange
import numpy as np
import cv2
 
def test(trainer, model, args: argparse.Namespace) -> None:
    load_ckps = False
    if args.mode.lower() != 'train' and model is None:
        assert args.ckpts_path is not None, "Checkpoint path must be provided for testing"
        assert os.path.exists(args.ckpts_path), f"Checkpoint path {args.ckpts_path} does not exist"
        print("Testing model from checkpoint", args.ckpts_path)
        load_ckps = True

    subset = 'test'
    target_size = (args.img_height, args.img_width)

    device = torch.device(f"cuda:{str(get_rank())}" if torch.cuda.is_available() else "cpu")
    args.accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

    testset = get_dataset(dataset=args.dataset, 
                        data_path=args.data_path, 
                        target_size=target_size, 
                        num_classes=args.num_classes,
                        subset=subset, 
                        seed=args.seed, 
                        split=args.split,
                        k_folds=args.k_folds,
                        no_validation=args.no_validation,
                        )

    if args.num_classes != testset.num_classes:
        raise ValueError(f"Number of classes in config file ({args.num_classes}) does not match the number of classes in the dataset ({testset.num_classes})")
    
    test_loader = get_dataloader(testset, target_size, args.batch_size, False, subset, None, args.num_workers, args.persistent_workers, args.pin_memory, device)
    
    net_kwargs = extract_net_params(args)
    optim_kwargs = extract_optim_params(args)

    net_kwargs['classes'] = testset.classes
    optim_kwargs['dataset_length'] = len(test_loader) * args.batch_size

    loss = nn.CrossEntropyLoss(reduction="mean", 
                               label_smoothing=args.label_smoothing, 
                               weight=torch.tensor(args.class_weights).to(device) if args.class_weights is not None else None
                               )
    model = select_arch(net_kwargs, loss, optim_kwargs)

    loggers = []
    if args.tensorboard:
        logger = TensorBoardLogger(os.path.dirname(args.run_path), name=args.experiment_name)
        loggers.append(logger)

    if args.wandb:
        assert os.environ["WANDB_API_KEY"] != "" and os.environ["WANDB_API_KEY"] is not None

        api_key = os.environ["WANDB_API_KEY"]
        wandb.login(key=api_key)

        wandb_logger = WandbLogger(name=args.experiment_name, save_dir=os.path.dirname(args.run_path), project="TakuNet")
        loggers.append(wandb_logger)

    if is_main_process():
        model.eval()
        inspection = summary(model, input_size=(args.batch_size, args.input_channels, *target_size), mode="eval")
        logging.info(f"is cuda available? {torch.cuda.is_available()}")
        logging.info(f"Using device {device}")
        logging.info("\n")

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name().lower()
            high_precision_gpus = ["a100", "a30", "a40", "a5000", "a6000", "3090", "6000", "5000", "4070", "orin"]
            if any([gpu.lower() in gpu_name for gpu in high_precision_gpus]):
                logging.info("Setting float32_matmul_precision to highest")
                torch.set_float32_matmul_precision("high")

        logging.info(inspection)
        logging.info("\n")

        inputs = torch.rand((1, 3, args.img_height, args.img_width)).to(device)
        flops = FlopCountAnalysis(model, inputs)
        print("FLOPs:", flops.total())
        logging.info(f"FLOPs: {flops.total()}")

    if trainer is None:
        trainer = L.Trainer(max_epochs=1, 
                        devices=[device.index] if torch.cuda.is_available() else 1,
                        strategy="auto",
                        accelerator=args.accelerator,
                        precision=args.lightning_precision,
                        deterministic=True,
                        logger=loggers,
                        )
    
    ckpts_file = os.path.join(args.ckpts_path, os.listdir(args.ckpts_path)[0]) if not args.ckpts_path.endswith(".ckpt") else args.ckpts_path
    trainer.test(model, 
                 test_loader, 
                 ckpt_path=ckpts_file if load_ckps else None
                 )
    