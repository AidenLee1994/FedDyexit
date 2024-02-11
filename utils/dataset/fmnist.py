import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "../")))
sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))

from utils.path_utils import getroot_path
from torchvision import transforms,datasets

def load_fmnist(dataset_name="fmnist",data_base_path=None):
    if data_base_path is None:
        data_base_path = getroot_path() + "data/"

    apply_transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor()])
    train_dataset = datasets.FashionMNIST(data_base_path + dataset_name + "/", train=True, download=True,
                                          transform=apply_transform)
    test_dataset = datasets.FashionMNIST(data_base_path + dataset_name + "/", train=False, download=True,
                                         transform=apply_transform)
    
    return train_dataset, test_dataset