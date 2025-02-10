import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
from timm.utils import ModelEma

import matplotlib.pyplot as plt
from utils.optimizer import get_optimizer
from torchmetrics.classification import Accuracy, F1Score, MulticlassConfusionMatrix
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR, LambdaLR, StepLR

import timeit
import logging

class LightningNet(L.LightningModule):
    def __init__(self, 
                 net_kwargs: dict,
                 criterion: nn.Module,
                 optim_kwargs: dict,
                 ): 
        super(LightningNet, self).__init__()
        self.net_kwargs = net_kwargs
        self.optim_kwargs = optim_kwargs
        self.criterion = criterion

        self.batch_size = optim_kwargs['batch_size']
        self.num_epochs = optim_kwargs['num_epochs']
        self.dataset_length = optim_kwargs['dataset_length']
        self.classes = net_kwargs['classes']
        self.output_classes = net_kwargs['output_classes']
        assert len(self.classes) == self.output_classes, "Number of classes does not match the output classes"

        self.optimizer = optim_kwargs['optimizer']
        self.scheduler = optim_kwargs['scheduler']
        self.scheduler_per_epoch = optim_kwargs['scheduler_per_epoch']
        self.learning_rate = optim_kwargs['learning_rate']
        self.learning_rate_decay = optim_kwargs['learning_rate_decay']
        self.learning_rate_decay_steps = optim_kwargs['learning_rate_decay_steps']
        self.min_learning_rate = optim_kwargs['min_learning_rate']
        self.warmup_epochs = optim_kwargs['warmup_epochs']
        self.warmup_steps = optim_kwargs['warmup_steps']
        self.weight_decay = optim_kwargs['weight_decay']
        self.weight_decay_end = optim_kwargs['weight_decay_end']
        self.update_freq = optim_kwargs['update_freq']
        self.alpha = optim_kwargs['alpha']
        self.momentum = optim_kwargs['momentum']

        self.max_batch_idx = self.dataset_length // self.batch_size

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.best_accuracy = 0.0
        self.best_f1 = 0.0
        self.fps_list = []

        self.getAccuracy = Accuracy(task='multiclass', num_classes=self.output_classes, average='weighted')
        self.getAccuracyTopX = Accuracy(task='multiclass', num_classes=self.output_classes, average='weighted', top_k=2)
        self.getF1Score = F1Score(task='multiclass', num_classes=self.output_classes, average='weighted')
        self.getConfusionMatrix = MulticlassConfusionMatrix(num_classes=self.output_classes, normalize='pred')

        if optim_kwargs['model_ema']:
            self.model_ema =  ModelEma(
            self,
            decay=0.9999,
            resume='')
        else:
            self.model_ema = None
    
    def override_device(self, device):
        self.getAccuracy.to(device)
        self.getAccuracyTopX.to(device)
        self.getF1Score.to(device)
        self.getConfusionMatrix.to(device)

    def on_train_epoch_start(self):
        self.training_step_outputs = []
        if hasattr(self.trainer.train_dataloader.dataset, 'k_folds'):
            if self.trainer.train_dataloader.dataset.k_folds > 0:
                if self.current_epoch != 0:
                    self.k_folds = self.trainer.train_dataloader.dataset.k_folds
                    self.current_fold = self.trainer.train_dataloader.dataset.current_fold
                    train_indices = self.trainer.train_dataloader.dataset.folds[self.current_fold][0].tolist()
                    val_indices = self.trainer.train_dataloader.dataset.folds[self.current_fold][1].tolist()

                    self.trainer.train_dataloader.dataset.indices = train_indices
                    self.trainer.val_dataloaders.dataset.indices = val_indices

    def on_validation_epoch_start(self):
        self.validation_step_outputs = []

    def on_test_epoch_start(self):
        self.test_step_outputs = []

    def training_step(self, batch: dict, batch_idx: int):
        images, labels = batch

        outputs = self(images)
        scores = F.softmax(outputs, dim=1)
        loss = self.criterion(outputs, labels)

        self.log('train/running_loss', loss, prog_bar=True, logger=True)

        output = {"loss": loss, 
                  "learning_rate": float(self.optimizer.param_groups[0]['lr']),
                  "true_labels": labels.cpu().detach(),
                  "pred_labels": scores.cpu().detach(),
                  }
        
        self.training_step_outputs.append(output)

        return output 
    

    def validation_step(self, batch, batch_idx):
        assert not self.training, "Model is still in training mode!"

        images, labels = batch
        outputs = self(images)
        scores = F.softmax(outputs, dim=1)
        loss = self.criterion(outputs, labels)

        self.override_device(self.device)
        self.log('val/loss_ckpts', loss, prog_bar=True)

        output = {"loss": loss, 
                  "true_labels": labels.cpu().detach(),
                  "pred_labels": scores.cpu().detach(),
                  }
        
        self.validation_step_outputs.append(output)
        
        return output
    

    def test_step(self, batch, batch_idx):
        images, labels = batch
        
        self.fps_output = None        
        if torch.cuda.is_available():
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)

            start_time.record()
            outputs = self(images)
            end_time.record()

            torch.cuda.synchronize()
            elapsed_time = start_time.elapsed_time(end_time) / 1000 # ms to s
            self.fps_list.append(1/elapsed_time) # this has not been used for fps logging
        else:
            def run_inference():
                self.fps_output = self(images)
                
            elapsed_time = timeit.timeit(run_inference, number=1)
            self.fps_list.append(1/elapsed_time)

        outputs = self.fps_output if self.fps_output is not None else outputs
        scores = F.softmax(outputs, dim=1)
        loss = self.criterion(outputs, labels)

        output = {"loss": loss, 
                  "true_labels": labels.cpu().detach(),
                  "pred_labels": scores.cpu().detach(),
                  "images": images.cpu().detach(),
                  }
        
        self.test_step_outputs.append(output)
        
        return output
    
    def configure_optimizers(self):
        self.optimizer = get_optimizer(
            self, self.optimizer, self.learning_rate, self.weight_decay,
            alpha=self.alpha, momentum=self.momentum
        )
        logging.info(f"Optimizer: {self.optimizer}")
        
        output = {'optimizer': self.optimizer}
        
        if self.scheduler is not None:
            if self.scheduler == 'cosine':
                scheduler = CosineAnnealingLR(self.optimizer, T_max=self.num_epochs, eta_min=self.min_learning_rate if self.min_learning_rate is not None else 0)
            elif self.scheduler == 'cyclic':
                scheduler = CyclicLR(
                    self.optimizer, base_lr=self.min_learning_rate,
                    max_lr=self.learning_rate, step_size_up=self.max_batch_idx,
                    cycle_momentum=False
                )
            elif self.scheduler == 'step':
                scheduler = StepLR(
                    self.optimizer, step_size=self.learning_rate_decay_steps,
                    gamma=self.learning_rate_decay
                )
            elif self.scheduler == 'lambda':
                scheduler = LambdaLR(
                    self.optimizer, lr_lambda=lambda epoch: 1 - epoch / self.num_epochs
                )
            else:
                raise ValueError(f"Unsupported scheduler: {self.scheduler}")

            logging.info(f"Scheduler: {scheduler}")
            output['lr_scheduler'] = {
                'scheduler': scheduler,
                'interval': 'epoch' if self.scheduler_per_epoch else 'step',
                'frequency': self.update_freq if self.update_freq else 1,
            }

            logging.info(f"Scheduler: {output['lr_scheduler']}")

        return output
    
    def on_before_zero_grad(self, *args, **kwargs):
        if hasattr(self, 'model_ema') and self.model_ema is not None:
            self.model_ema.update(self)

    def on_train_epoch_end(self):
        self.base_logger(self.training_step_outputs, mode='train')        
        self.training_step_outputs.clear()

        if hasattr(self.trainer.train_dataloader.dataset, 'k_folds'):
            if self.trainer.train_dataloader.dataset.k_folds > 0:
                self.current_fold = (self.current_fold + 1) % self.k_folds if self.current_epoch > 0 else 0
                self.trainer.train_dataloader.dataset.set_kfold(self.current_fold)

    def on_validation_epoch_end(self):
        acc1, acc2, f1 = self.base_logger(self.validation_step_outputs, mode='val')
        
        self.best_accuracy = max(self.best_accuracy, acc1.item())
        self.best_f1 = max(self.best_f1, f1)
        self.logger.experiment.add_scalar('val/best_acc1', self.best_accuracy, self.current_epoch)
        self.logger.experiment.add_scalar('val/best_f1', self.best_f1, self.current_epoch)
        logging.info(f"Best accuracy so far: {self.best_accuracy}")

        self.confusion_matrix(self.validation_step_outputs, mode='val')

        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        acc1, acc2, f1 = self.base_logger(self.test_step_outputs, mode='test')
        
        self.best_accuracy = max(self.best_accuracy, acc1.item())
        self.best_f1 = max(self.best_f1, f1)
        self.logger.experiment.add_scalar('test/best_acc1', self.best_accuracy, self.current_epoch)
        self.logger.experiment.add_scalar('test/best_f1', self.best_f1, self.current_epoch)
        logging.info(f"Accuracy on Test Set: {self.best_accuracy}")
        logging.info(f"F1 Score on Test Set: {self.best_f1}")

        self.confusion_matrix(self.test_step_outputs, mode='test')

        fps_mean = torch.tensor(self.fps_list).div(self.batch_size).mean().item()
        self.logger.experiment.add_scalar('test/fps', fps_mean, self.current_epoch)
        logging.info(f"Mean FPS: {fps_mean}")

        self.test_step_outputs.clear()

    def base_logger(self, step_outputs, mode: str=['train', 'val', 'test']):
        assert mode in ['train', 'val', 'test'], "Mode must be one of ['train', 'val', 'test']"
        self.override_device(self.device)

        loss = torch.stack([x['loss'] for x in step_outputs]).mean().item()
        true_labels = torch.cat([x['true_labels'] for x in step_outputs], dim=0).view(-1, 1).to(self.device)
        pred_labels = torch.cat([x['pred_labels'] for x in step_outputs], dim=0).view(-1, self.output_classes).to(self.device)

        acc1 = self.getAccuracy(pred_labels, true_labels.view(-1))
        accX = self.getAccuracyTopX(pred_labels, true_labels.view(-1))
        f1 = self.getF1Score(pred_labels, true_labels.view(-1))

        if mode == 'train':
            learning_rate = torch.tensor([x['learning_rate'] for x in step_outputs]).mean().item()
            self.logger.experiment.add_scalar(f'{mode}/learning_rate', learning_rate, self.current_epoch)

        self.logger.experiment.add_scalar(f'{mode}/loss', loss, self.current_epoch)
        self.logger.experiment.add_scalar(f'{mode}/acc1', acc1.item(), self.current_epoch)
        self.logger.experiment.add_scalar(f'{mode}/acc{self.getAccuracyTopX.top_k}', accX.item(), self.current_epoch)
        
        return acc1, accX, f1
    
    def confusion_matrix(self, step_outputs, mode: str=['train', 'val', 'test']):
        assert mode in ['train', 'val', 'test'], "Mode must be one of ['train', 'val', 'test']"
        self.override_device(self.device)

        true_labels = torch.cat([x['true_labels'] for x in step_outputs], dim=0).view(-1, 1).to(self.device)
        pred_labels = torch.cat([x['pred_labels'] for x in step_outputs], dim=0).view(-1, self.output_classes).to(self.device)

        self.getConfusionMatrix.update(pred_labels, true_labels.view(-1))
        fig, ax = self.getConfusionMatrix.plot(labels=self.classes)

        self.logger.experiment.add_figure(f'{mode}/confusion matrix', fig, self.current_epoch)
        plt.close(fig)

    


