import torch
from PIL.Image import Image
from torch import nn, optim, cuda
from torch.utils.data import DataLoader

from torchvision import transforms
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights

from dataset_preprocessing import calculate_normalization, create_dataset
from constants import PATH_TRAIN, PATH_TEST, PATH_VALID, NORMALIZATION_PARAMS, LR, NUM_EPOCHS

# def unnormalize_and_convert_to_pil(tensor):
#
#     mean = NORMALIZATION_PARAMS['mean']
#     std = NORMALIZATION_PARAMS['std']
#     unnormalize = transforms.Normalize(
#         mean=[-m / s for m, s in zip(mean, std)],  # Reverse the normalization
#         std=[1 / s for s in std]
#     )
#
#     # Reverse normalization
#     tensor = unnormalize(tensor)
#     # Convert tensor to PIL Image
#     to_pil = transforms.ToPILImage()
#     return to_pil(tensor)


if __name__ == '__main__':

    # PREPROCESSING

    # Calculation of custom mean and std values for more accurate normalization
    dataset_transformation = NORMALIZATION_PARAMS

    # Initializing a training, validation and testing set
    train_set = create_dataset(PATH_TRAIN, dataset_transformation)
    valid_set = create_dataset(PATH_VALID, dataset_transformation)
    test_set = create_dataset(PATH_TEST, dataset_transformation)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_set, batch_size=16, shuffle=True, num_workers=2)

    print(test_set.classes)

    # CNN DEPLOYING

    my_efficientB0 = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    classes_num = len(train_set.classes)
    print(classes_num)

    my_efficientB0.classifier[1] = nn.Linear(my_efficientB0.classifier[1].in_features, classes_num)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(my_efficientB0.parameters(), lr=LR)

    device = 'cuda' if cuda.is_available() else 'cpu'

    my_efficientB0 = my_efficientB0.to(device)

    # TRAINING AND VALIDATING
    for epoch in range(NUM_EPOCHS):

        my_efficientB0.train()
        running_loss = 0.0
        correct = 0
        total = 0
        process = 0

        for inputs, labels in train_loader:
            process += 1
            print(f"{epoch + 1} epoch processing training... {process}")
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = my_efficientB0(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        print(f"Epoch #{epoch + 1}/10 ------- Loss: {running_loss / len(train_loader):4f}, "
              f"Accuracy: {100 * correct / total:2f}%")

        my_efficientB0.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = my_efficientB0(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        print(f"Validation Loss: {val_loss / len(valid_loader):.4f}, Accuracy: {100 * val_correct / val_total:.2f}%")

    # TESTING

    my_efficientB0 = my_efficientB0.to(device)

    img_tensor, actual_label = test_set[249]
    to_pil = transforms.ToPILImage()
    image_to_test = to_pil(img_tensor)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = my_efficientB0(img_tensor)

    _, predicted_class = torch.max(outputs.data, 1)
    predicted_class_name = test_set.classes[predicted_class.item()]

    print(f"Actual: {test_set.classes[actual_label]}")
    print(f"Predicted: {predicted_class_name}")
    image_to_test.show()
