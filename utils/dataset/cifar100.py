import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "../")))
sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))

from utils.path_utils import getroot_path
from torchvision import transforms,datasets

def load_cifar100(dataset_name="cifar100",data_base_path=None):
    if data_base_path is None:
        data_base_path = getroot_path() + "data/"
    mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),  # 数据增强
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    train_dataset = datasets.CIFAR100(data_base_path + dataset_name + "/", train=True, download=True,
                                     transform=train_transform)
    test_dataset = datasets.CIFAR100(data_base_path + dataset_name + "/", train=False, download=True,
                                    transform=test_transform)
    
    return train_dataset, test_dataset

if __name__ == '__main__':
    load_cifar100()