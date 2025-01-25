import os
import random
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import datasets, transforms
from constants import BATCH_SIZE as BS, RESIZE
from constants import HORFLIP, CROP, ROTATION


def calculate_normalization(train_path):

    # Non-normalized transformation
    non_normalized_transform = transforms.Compose([
        transforms.Resize((RESIZE, RESIZE)),
        transforms.ToTensor()
    ])

    # Initializing a whole training set instance
    train_set = datasets.ImageFolder(train_path, non_normalized_transform)
    train_loader = DataLoader(train_set, BS, False)

    # Calculating dataset mean and std values
    mean = 0.0
    std = 0.0
    dataset_size = 0

    for images, _ in train_loader:
        batch_size = images.size(0)
        dataset_size += batch_size
        print("Images preprocessed: ", dataset_size)
        print("Mean: ", mean, "STD: ", std)
        mean += images.mean([0, 2, 3]) * batch_size
        std += images.std([0, 2, 3]) * batch_size

    mean /= dataset_size
    std /= dataset_size

    print("Final values: ( ", mean, ", ", std, ")")

    my_normalization = transforms.Normalize(mean, std)

    # Making a transformation with specified normalization values
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        my_normalization
    ])

    return train_transform


def augment_dataset(path_dir, threshold):
    diseases = os.listdir(path_dir)

    for disease_dir in diseases:
        dis1 = os.path.join(path_dir, diseases[0])
        print(os.listdir(dis1))
    pass


def create_dataset(dir_path, transforms):
    return datasets.ImageFolder(dir_path, transforms)
