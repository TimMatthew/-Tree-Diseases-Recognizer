import torch
from torch import nn, optim, device, cuda
from torch.utils.data import DataLoader
from dataset_preprocessing import calculate_normalization, create_dataset
from constants import PATH_TRAIN, PATH_TEST, PATH_VALID, NORMALIZATION_PARAMS
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights


def main():
    print("Hello World")


if __name__ == '__main__':
    # PREPROCESSING

    print(torch.__version__)
    # Calculation of custom mean and std values for more accurate normalization
    dataset_transformation = NORMALIZATION_PARAMS

    # Initializing a training, validation and testing set
    train_set = create_dataset(PATH_TRAIN, dataset_transformation)
    valid_set = create_dataset(PATH_VALID, dataset_transformation)
    test_set = create_dataset(PATH_TEST, dataset_transformation)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_set, batch_size=16, shuffle=True, num_workers=2)

    # CNN DEPLOYING

    my_efficientB0 = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    classes_num = len(train_set.classes)

    my_efficientB0.classifier[1] = nn.Linear(my_efficientB0.classifier[1].in_features, classes_num)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(my_efficientB0.parameters(), lr=0.001)

    if cuda.is_available():
        device = device("cuda")
        print(device)

    # my_efficientB0 = my_efficientB0.to(device)
    #
    # epochs = 10
    #
    # for epoch in range(epochs):
    #     my_efficientB0.train()
    #     running_loss = 0.0
    #     correct = 0
    #     total = 0
    #     print("epoch")
    #
    #     for inputs, labels in train_loader:
    #
    #         print("processing training...")
    #         inputs, labels = inputs.to(device), labels.to(device)
    #
    #         optimizer.zero_grad()
    #
    #         outputs = my_efficientB0(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #
    #         running_loss += loss.item()
    #         _, predicted = outputs.max(1)
    #         total += labels.size(0)
    #         correct += predicted.eq(labels).sum().item()
    #
    #     print(f"Epoch #{epoch+1}/10 ------- Loss: {running_loss/len(train_loader):4f}, "
    #           f"Accuracy: {100*correct/total:2f}%")
    # model2 = models.efficientnet_b4(pretrained=True)
