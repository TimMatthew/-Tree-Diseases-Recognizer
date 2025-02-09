from torchvision import datasets
from torch import float32
from torchvision import datasets
from torchvision.transforms import v2

from constants import RESIZE_B3


def calculate_normalization(train_path):
    # Non-normalized transformation
    non_normalized_transform = v2.Compose([
        v2.Resize((RESIZE_B3, RESIZE_B3)),
        v2.ToImage(),
        v2.ToDtype(float32)
    ])


def create_dataset(dir_path, transforms):
    return datasets.ImageFolder(dir_path, transforms)
