from typing import Tuple
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torch.utils.data import random_split

import torch
import torchvision.datasets as datasets

VAL_RATIO = 0.1

# Maintain consistent dataset split
torch.manual_seed(0)

# Data augmentation transform fro vgg models
def get_basic_transform() -> transforms:
    return transforms.Compose([
        # Data augmentation
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness = 0.4, contrast = 0.4, saturation = 0.4, hue = 0.1),
        transforms.ToTensor(),

        # data-standardization for imgnet dataset
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ])

def get_datasets(
    train_root_dir: str,
    transforms: transforms,
    val_ratio = VAL_RATIO
) -> Tuple[Dataset, Dataset, int]:
    dataset = datasets.ImageFolder(
        root = train_root_dir,
        transform = transforms
    )

    train_set, val_set = random_split(dataset, [1 - val_ratio, val_ratio])

    return train_set, val_set, len(dataset)