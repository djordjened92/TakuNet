import numpy as np
import torch
import torchvision
from torchvision.datasets import DatasetFolder
from sklearn.model_selection import StratifiedKFold
import logging


class AIDER(DatasetFolder):
    PROPORTIONAL_SPLITS = {
        "collapsed_building": (335, 30, 146),
        "fire": (343, 30, 148),
        "flooded_areas": (346, 30, 150),
        "normal": (2450, 400, 1540),
        "traffic_incident": (316, 30, 139),
    }

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
                 subset: str, 
                 seed: int, 
                 split: str = '',
                 k_folds: int = 0,
                 no_validation: bool = False) -> None:
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
            self.indices = self.KFold_split(subset, self.splits, self.k_folds, seed)
        else:
            logging.info("Using stratified split method.")
            self.indices = self.stratified_datasplit(subset, self.splits, self.no_validation)

        logging.info(f"Dataset {subset} size: {len(self.indices)}")

    def loader(self, path):
        return torchvision.io.read_image(path)

    def __getitem__(self, index):
        actual_index = self.indices[index]
        img, label = super().__getitem__(actual_index)
        return img, label

    def __len__(self):
        return len(self.indices)

    def stratified_datasplit(self, subset: str, splits: dict, no_validation: bool = False) -> list:
        if no_validation:
            logging.info("Fusing validation set with training set.")

        train_indices = []
        val_indices = []
        test_indices = []

        targets = torch.tensor(self.targets)
        for cls in self.classes:
            cls_idx = self.class_to_idx[cls]
            assert sum(splits[cls]) == targets[targets == cls_idx].numel(), (
                f"Class {cls} has {targets[targets == cls_idx].numel()} samples, "
                f"but the splits sum to {sum(splits[cls])}"
            )

            cls_indices = np.where(targets == cls_idx)[0]
            np.random.shuffle(cls_indices)
            train_samples, val_samples, test_samples = splits[cls]

            train_idx = cls_indices[:train_samples + val_samples] if no_validation else cls_indices[:train_samples]
            val_idx = cls_indices[train_samples:train_samples + val_samples]
            test_idx = cls_indices[train_samples + val_samples:]

            train_indices.extend(train_idx)
            val_indices.extend(val_idx)
            test_indices.extend(test_idx)

        if subset == 'train':
            return train_indices
        elif subset == 'val':
            return val_indices
        elif subset == 'test':
            return test_indices
        else:
            raise ValueError(f"Unknown subset: {subset}")

    def KFold_split(self, subset: str, splits: dict, k: int, seed: int) -> list:
        trainval_indices = []
        test_indices = []

        targets = torch.tensor(self.targets)
        for cls in self.classes:
            cls_idx = self.class_to_idx[cls]
            assert sum(splits[cls]) == targets[targets == cls_idx].numel(), (
                f"Class {cls} has {targets[targets == cls_idx].numel()} samples, "
                f"but the splits sum to {sum(splits[cls])}"
            )

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
        if subset == 'train':
            return self.folds[self.current_fold][0].tolist()
        elif subset == 'val':
            return self.folds[self.current_fold][1].tolist()
        elif subset == 'test':
            return test_indices
        else:
            raise ValueError(f"Unknown subset: {subset}")

    def set_kfold(self, k: int):
        assert self.k_folds > 0, "This function is only available when k_folds > 0"
        self.current_fold = k
