from augmentation.aider import create_aider_augmentations

from albumentations import Compose as albu_compose

def select_augmentation(aug_type: str, target_size: tuple, p: float) -> albu_compose:
    if aug_type is None:
        return None
    if aug_type.upper() == 'AIDER':
        return create_aider_augmentations(*target_size, p)
    else:
        raise ValueError(f"Augmentation type {aug_type} not supported")