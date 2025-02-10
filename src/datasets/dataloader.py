import numpy as np
import torch
import torchvision
from torchvision.datasets import DatasetFolder, FakeData
from torch.utils.data import DataLoader
from albumentations import Compose as AlbuCompose
from torchvision.transforms import Compose as TorchCompose

from datasets.aider import AIDER
from datasets.aiderv2 import AIDERV2

class CollateFnWrapper:
    def __init__(self, target_size: tuple, subset: str, transforms: AlbuCompose, device: torch.device):
        self.device = device
        self.transforms = transforms
        self.subset = subset

        self.resize = torchvision.transforms.Resize(target_size)

    def __call__(self, batch):
        images, labels = zip(*batch)

        if self.subset == 'train' and self.transforms is not None:
            if isinstance(self.transforms, AlbuCompose):
                images = np.stack([self.transforms(image=img.permute(1, 2, 0).detach().cpu().numpy())["image"] for img in images])
                images = torch.from_numpy(np.transpose(images, (0, 3, 1, 2))).to(torch.float32).to(self.device)
            elif isinstance(self.transforms, TorchCompose):
                images = [self.transforms(img) for img in images]
                images = torch.stack(images).to(self.device).to(torch.float32)
        else:
            images = [self.resize(img) for img in images]
            images = torch.stack(images).to(self.device).to(torch.float32)
        labels = torch.tensor(list(labels), device=self.device)

        images = images / 255.
        return images, labels

def get_dataloader(dataset: DatasetFolder, target_size: tuple, batch_size: int, shuffle: bool, subset: str, transforms: AlbuCompose, num_workers: int, persistent_workers: bool, pin_memory: bool, device: torch.device):
    collate_fn = CollateFnWrapper(target_size, subset=subset, transforms=transforms, device=device)
    dataloader = DataLoader(dataset, \
                            batch_size=batch_size, \
                            shuffle=shuffle, \
                            num_workers=num_workers, \
                            persistent_workers=persistent_workers, \
                            pin_memory=pin_memory, \
                            collate_fn=collate_fn)
    return dataloader
    
def get_dataset(dataset: str, 
                data_path: str, 
                target_size: tuple, 
                num_classes: int,
                subset:str, 
                seed: int,
                split: str='',
                k_folds: int=0,
                no_validation: bool=False,
                ) -> DatasetFolder:
    if dataset.upper() == "AIDER":
        return AIDER(data_path, target_size, subset, seed, split, k_folds, no_validation)
    elif dataset.upper() == "AIDERV2":
        return AIDERV2(data_path, target_size, subset)
    elif dataset.upper() == "FAKEDATA": # just for testing purposes
        return FakeData(size=100, image_size=(3, *target_size), num_classes=num_classes, transform=torchvision.transforms.ToTensor())
    else:
        raise ValueError(f"Dataset {dataset} not found.")