import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "../")))
sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))

from utils.path_utils import getroot_path
from torchvision import transforms,datasets

def load_cifar10(dataset_name="cifar10",data_base_path=None):
    if data_base_path is None:
        data_base_path = getroot_path() + "data/"

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    train_dataset = datasets.CIFAR10(data_base_path + dataset_name + "/", train=True, download=True,
                                     transform=train_transform)
    test_dataset = datasets.CIFAR10(data_base_path + dataset_name + "/", train=False, download=True,
                                    transform=test_transform)
    
    return train_dataset, test_dataset