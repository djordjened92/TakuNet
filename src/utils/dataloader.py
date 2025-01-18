import os
import numpy as np
import torch
import torchvision
from torchvision.datasets import DatasetFolder, FakeData
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import StratifiedKFold

from albumentations import Compose
import logging

class CollateFnWrapper:
    def __init__(self, target_size: tuple, subset: str, transforms: Compose, device: torch.device):
        self.device = device
        self.transforms = transforms
        self.subset = subset

        self.resize = torchvision.transforms.Resize(target_size)

    def __call__(self, batch):
        images, labels = zip(*batch)

        if self.subset == 'train' and self.transforms is not None:
            images = np.stack([self.transforms(image=img.permute(1, 2, 0).detach().cpu().numpy())["image"] for img in images])
            images = torch.from_numpy(np.transpose(images, (0, 3, 1, 2))).to(torch.float32).to(self.device)
        else:
            images = [self.resize(img) for img in images]
            images = torch.stack(images).to(self.device).to(torch.float32)
        labels = torch.tensor(list(labels), device=self.device)

        images = images / 255.
        return images, labels

class AIDER(DatasetFolder):
    # class: (train samples, val samples, test samples)
    PROPORTIONAL_SPLITS = {
        "collapsed_building": (335, 30, 146),
        "fire": (343, 30, 148),
        "flooded_areas": (346, 30, 150),
        "normal": (2450, 400, 1540),
        "traffic_incident": (316, 30, 139),
    }

    # same splits as in TinyEmergencyNet and EmergencyNet
    EXACT_SPLITS = {
        "collapsed_building": (286, 25, 200),
        "fire": (281, 30, 210),
        "flooded_areas": (286, 40, 200),
        "normal": (2000, 390, 2000),
        "traffic_incident": (275, 10, 200),
    }

    def __init__(self, 
                 data_path: str, 
                 target_size: tuple, 
                 subset:str, 
                 seed: int, 
                 split: str='',
                 k_folds: int = 0,
                 no_validation: bool = False
                 ) -> None:
        super().__init__(data_path, loader=self.loader, extensions=('.jpg'))
        self.target_size = target_size
        self.subset = subset
        self.num_classes = len(self.classes)
        self.split = split
        self.k_folds = k_folds
        self.seed = seed
        self.no_validation = no_validation

        if split.lower() == 'proportional':
            logging.info("Using proportional split.")
            self.splits = self.PROPORTIONAL_SPLITS
        elif split.lower() == 'exact':
            logging.info("Using exact split.")
            self.splits = self.EXACT_SPLITS
        else:
            raise ValueError(f"Split type {split} not supported.")

        if self.k_folds > 0:
            logging.info(f"Using K-Fold split method with {self.k_folds} folds.")
            self.dataset = self.KFold_split(subset, self.splits, self.k_folds, seed)
        else:
            logging.info("Using stratified split method.")
            self.dataset = self.stratified_datasplit(subset, self.splits, self.no_validation)
        
        logging.info(f"Dataset {subset} size: {len(self.dataset)}")
        
    def loader(self, path):
        return torchvision.io.read_image(path)
    
    def __getitem__(self, index):
        return super().__getitem__(index)
    
    def __len__(self):
        return super().__len__()
    
    def get_weights(self, device: torch.device) -> torch.Tensor:
        assert self.dataset is not None, "Dataset is not initialized."
        weights = [value[0] for value in self.splits.values()]
        return (1 - (torch.tensor(weights) / sum(weights))).to(device)
    
    def stratified_datasplit(self, subset: str, splits: dict, no_validation: bool=False) -> Subset:    
        if no_validation:
            logging.info("Fusing validation set with training set.")

        train_indices = []
        val_indices = []
        test_indices = []
    
        targets = torch.tensor(self.targets)
        for cls in self.classes:
            cls_idx = self.class_to_idx[cls]
            assert sum(splits[cls]) == targets[targets == cls_idx].numel(), f"Class {cls} has {targets[targets == cls_idx].numel()} samples, but the splits sum to {sum(splits[cls])}"

            cls_indices = np.where(targets == cls_idx)[0]
            np.random.shuffle(cls_indices)
            train_samples, val_samples, test_samples = splits[cls]

            train_idx = cls_indices[:train_samples + val_samples] if no_validation else cls_indices[:train_samples] 
            val_idx = cls_indices[train_samples:train_samples + val_samples]
            test_idx = cls_indices[train_samples + val_samples:]
            
            train_indices.extend(train_idx)
            val_indices.extend(val_idx)
            test_indices.extend(test_idx)
        
        train_set = Subset(self, train_indices)
        val_set = Subset(self, val_indices)
        test_set = Subset(self, test_indices)

        if subset == 'train' or subset == 'test':
            self.log_class_distribution(train_set, val_set, test_set)

        if subset == 'train':
            return train_set
        elif subset == 'val':
            return val_set
        elif subset == 'test':
            return test_set
        else:
            raise ValueError(f"Unknown subset: {subset}")

    def KFold_split(self, subset: str, splits: dict, k: int, seed: int) -> Subset:
        trainval_indices = []
        test_indices = []
    
        targets = torch.tensor(self.targets)
        for cls in self.classes:
            cls_idx = self.class_to_idx[cls]
            assert sum(splits[cls]) == targets[targets == cls_idx].numel(), f"Class {cls} has {targets[targets == cls_idx].numel()} samples, but the splits sum to {sum(splits[cls])}"

            cls_indices = np.where(targets == cls_idx)[0]
            cls_indices = np.random.permutation(cls_indices)
            train_samples, val_samples, test_samples = splits[cls]

            trainval_idx = cls_indices[:train_samples + val_samples]
            test_idx = cls_indices[train_samples + val_samples:]
            
            trainval_indices.extend(trainval_idx)
            test_indices.extend(test_idx)

        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        self.folds = list(skf.split(np.array(trainval_indices), np.array(self.targets)[np.array(trainval_indices)]))

        self.current_fold = 0
        train_set = Subset(self, self.folds[self.current_fold][0].tolist())
        val_set = Subset(self, self.folds[self.current_fold][1].tolist())
        test_set = Subset(self, test_indices)

        if subset == 'train' or subset == 'test':
            self.log_class_distribution(train_set, val_set, test_set)

        if subset == 'train':
            return train_set
        elif subset == 'val':
            return val_set
        elif subset == 'test':
            return test_set
        else:
            raise ValueError(f"Unknown subset: {subset}")
        
    def set_kfold(self, k: int):
        assert self.k_folds > 0, "this function is only available when k_folds > 0"
        self.current_fold = k
    
    def log_class_distribution(self, train_set, val_set, test_set):
        train_counts = self.count_classes(train_set)
        val_counts = self.count_classes(val_set)
        test_counts = self.count_classes(test_set)

        logging.info("Class distribution:")
        logging.info(f"{'Class':<10} {'Train':<10} {'Validation':<10} {'Test':<10}")
        logging.info("-" * 40)
        logging.info("")

        for cls in range(self.num_classes):
            train_count = train_counts.get(cls, 0)
            val_count = val_counts.get(cls, 0)
            test_count = test_counts.get(cls, 0)
            print(f"{cls:<10} {train_count:<10} {val_count:<10} {test_count:<10}")
            logging.info(f"{cls:<10} {train_count:<10} {val_count:<10} {test_count:<10}")

    def count_classes(self, dataset):
            class_counts = {}
            for idx in dataset.indices:
                label = self.targets[idx]
                if label not in class_counts:
                    class_counts[label] = 0
                class_counts[label] += 1
            return class_counts
    

class AIDERV2(DatasetFolder):
    def __init__(self, data_path: str, target_size: tuple, subset:str):
        if subset not in ['train', 'val', 'test']:
            raise ValueError("subset must be 'train', 'val' or 'test.")
        subset = subset.capitalize()

        super().__init__(os.path.join(data_path, subset), loader=self.loader, extensions=('.png',))

        self.target_size = target_size
        self.subset = subset
        self.k_folds = 0

        self.num_classes = len(self.classes)
        self.dataset = self
        
    def loader(self, path):
        return torchvision.io.read_image(path)
    
    def __getitem__(self, index):
        return super().__getitem__(index)
    
    def __len__(self):
        return super().__len__()


def get_dataloader(dataset: DatasetFolder, target_size: tuple, batch_size: int, shuffle: bool, subset: str, transforms: Compose, num_workers: int, persistent_workers: bool, pin_memory: bool, device: torch.device):
    collate_fn = CollateFnWrapper(target_size, subset=subset, transforms=transforms, device=device)
    dataloader = DataLoader(dataset.dataset, \
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