import os
import torchvision
from torchvision.datasets import DatasetFolder

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