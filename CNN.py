import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import EfficientNet_B1_Weights, EfficientNet_B0_Weights
from constants import LR, NUM_EPOCHS
from torch import cuda


# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()

def deploy_cnn(train_set):
    efficientB0 = models.efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
    classes_num = len(train_set.classes)
    print(classes_num)

    efficientB0.classifier[1] = nn.Linear(efficientB0.classifier[1].in_features, classes_num)
    device = 'cuda' if cuda.is_available() else 'cpu'

    return efficientB0.to(device)


def train(model, train_loader, valid_loader):
    device = 'cuda' if cuda.is_available() else 'cpu'

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(NUM_EPOCHS):

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        process = 0

        for inputs, labels in train_loader:
            process += 1
            print(f"{epoch + 1} epoch processing training... {process}")
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        print(f"Epoch #{epoch + 1}/{NUM_EPOCHS} ------- Loss: {running_loss / len(train_loader):4f}, "
              f"Accuracy: {100 * correct / total:2f}%")

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

            print(f"Validation Loss: {val_loss / len(valid_loader):.4f}, Accuracy: {100 * val_correct / val_total:.2f}%")

    return model


def test(model, test_set, tensor_num):
    device = 'cuda' if cuda.is_available() else 'cpu'

    my_efficientB0 = model.to(device)

    img_tensor, actual_label = test_set[tensor_num]
    to_pil = transforms.ToPILImage()
    image_to_test = to_pil(img_tensor)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = my_efficientB0(img_tensor)

    _, predicted_class = torch.max(outputs.data, 1)
    predicted_class_name = test_set.classes[predicted_class.item()]

    print(f"Actual: {test_set.classes[actual_label]}")
    print(f"Predicted: {predicted_class_name}\n")
    image_to_test.show()
