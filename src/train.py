import argparse
import logging
import os
import wandb

import torch
import torch.nn as nn
from torchinfo import summary

import lightning as L
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from utils.distributed import get_rank, is_main_process
from utils.net_utils import extract_net_params, extract_optim_params
from fvcore.nn import FlopCountAnalysis

from utils.net_utils import select_arch
from augmentation.augmentator import select_augmentation
from datasets.dataloader import get_dataset, get_dataloader

def train(args: argparse.Namespace) -> None:
    device = torch.device(f"cuda:{str(get_rank())}" if torch.cuda.is_available() else "cpu")
    transforms = select_augmentation(args.augment, (args.img_height, args.img_width), p=0.1)
    target_size=(args.img_height, args.img_width)

    trainset = get_dataset(dataset=args.dataset, 
                           data_path=args.data_path, 
                           target_size=target_size, 
                           num_classes=args.num_classes,
                           subset='train', 
                           seed=args.seed,  
                           split=args.split,
                           k_folds=args.k_folds,
                           no_validation=args.no_validation,
                           )
    
    valset = get_dataset(dataset=args.dataset, 
                        data_path=args.data_path, 
                        target_size=target_size,  
                        num_classes=args.num_classes,
                        subset='val', 
                        seed=args.seed, 
                        split=args.split,
                        k_folds=args.k_folds,
                        no_validation=args.no_validation,
                        )

    if args.num_classes != trainset.num_classes:
        raise ValueError(f"Number of classes in config file ({args.num_classes}) does not match the number of classes in the dataset ({trainset.num_classes})")
    
    train_loader = get_dataloader(trainset, target_size, args.batch_size, True, 'train', transforms, args.num_workers, args.persistent_workers, args.pin_memory, device)
    val_loader = get_dataloader(valset, target_size, args.batch_size, False, 'val', transforms, args.num_workers, args.persistent_workers, args.pin_memory, device)

    net_kwargs = extract_net_params(args)
    optim_kwargs = extract_optim_params(args)
    
    net_kwargs['classes'] = trainset.classes
    optim_kwargs['dataset_length'] = len(train_loader) * args.batch_size

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
        inspection = summary(model, input_size=(args.batch_size, args.input_channels, *target_size), mode="train")
        logging.info(f"is cuda available? {torch.cuda.is_available()}")
        logging.info(f"Using device {device}")
        logging.info("\n")

        gpu_name = torch.cuda.get_device_name().lower()
        high_precision_gpus = ["a100", "a30", "a40", "a5000", "a6000", "3090", "6000", "5000", "4070"]
        if any([gpu.lower() in gpu_name for gpu in high_precision_gpus]):
            logging.info("Setting float32_matmul_precision to highest")
            torch.set_float32_matmul_precision("high")

        logging.info(inspection)
        logging.info("\n")

        flops = FlopCountAnalysis(model, torch.rand((1, 3, args.img_height, args.img_width)).to(device))
        print("FLOPs:", flops.total())
        logging.info(f"FLOPs: {flops.total()}")

    best_val_checkpoint = ModelCheckpoint(
        filename="best_model_val_loss",
        save_top_k=1,
        verbose=False,
        monitor="val/loss_ckpts",
        mode="min"
    )

    best_train_checkpoint = ModelCheckpoint(
        filename="best_model_train_loss",
        save_top_k=1,
        verbose=False,
        monitor="train/running_loss",
        mode="min"
    )

    last_model_checkpoint = ModelCheckpoint(
        filename="model-last",
        save_top_k=1,
        verbose=False,
        save_last=True,  
    )

    trainer = L.Trainer(max_epochs=args.num_epochs, 
                        devices=[device.index], 
                        strategy="auto",
                        precision=args.lightning_precision,
                        deterministic=True,
                        logger=loggers,
                        callbacks=[best_val_checkpoint, best_train_checkpoint, last_model_checkpoint],
                        )
    
    if is_main_process():
        logging.info(f"Initialized Trainer with {device}")
        logging.info(f"Using {args.lightning_precision} precision")
        logging.info(f"Training for {args.num_epochs} epochs ...\n")
    

    trainer.fit(model, 
                train_dataloaders=train_loader, 
                val_dataloaders=val_loader,
                )
    
    if is_main_process():
        logging.info("Training completed")

    return trainer, model
