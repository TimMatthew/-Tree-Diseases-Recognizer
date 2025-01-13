from torchvision import transforms
PATH_TRAIN = r'C:\Users\tymop\OneDrive\Робочий стіл\Курсова\dataset\train'
PATH_VALID = r'C:\Users\tymop\OneDrive\Робочий стіл\Курсова\dataset\valid'
PATH_TEST = r'C:\Users\tymop\OneDrive\Робочий стіл\Курсова\dataset\test'

# ALPHA = 0
OPTIMIZER = "Adam"
LR = 0.001
BATCH_SIZE = 32
RESIZE_FACTOR = 224
NORMALIZATION_PARAMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4523, 0.4814, 0.3338),  (0.2279, 0.2170, 0.2207))
])
