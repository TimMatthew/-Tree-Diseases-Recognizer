import torch
from PIL.Image import Image
from torch import nn, optim, cuda

from torch.utils.data import DataLoader

from torchvision import transforms
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights

from dataset_preprocessing import calculate_normalization, create_dataset, augment_dataset
from constants import PATH_TRAIN, PATH_TEST, PATH_VALID, NORMALIZATION_DEFAULT as ND

from CNN import deploy_cnn, train, test

if __name__ == '__main__':

    # PREPROCESSING

    # Calculation of custom mean and std values for more accurate normalization

    # Initializing a training, validation and testing set
    train_set = create_dataset(PATH_TRAIN, ND)
    # augmented_train_set = augment_dataset(PATH_TRAIN, 600)

    valid_set = create_dataset(PATH_VALID, ND)
    test_set = create_dataset(PATH_TEST, ND)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=6)
    valid_loader = DataLoader(valid_set, batch_size=16, shuffle=True, num_workers=6)

    print(test_set.classes)

    # CNN DEPLOYING
    conv_net = deploy_cnn(train_set)

    # TRAINING AND VALIDATING
    conv_net = train(conv_net, train_loader, valid_loader)

    # TESTING
    test(conv_net, test_set, 0)
    test(conv_net, test_set, 249)
    test(conv_net, test_set, 231)
    test(conv_net, test_set, 114)
    test(conv_net, test_set, 204)
    test(conv_net, test_set, 189)
    test(conv_net, test_set, 57)
    test(conv_net, test_set, 23)
    test(conv_net, test_set, 156)
