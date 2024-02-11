# -*- coding:utf-8 -*-
# date: 2023/5/23 下午3:57
import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../")))
sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))

import numpy as np
from torchvision import datasets,transforms
from torch.utils.data import Dataset
from torch import tensor
from utils.path_utils import getroot_path
from core.data_partition import paratition_data_index
from utils.dataset.pacs import load_pacs
from utils.dataset.mnist import load_mnist
from utils.dataset.fmnist import load_fmnist
from utils.dataset.cifar10 import load_cifar10
from utils.dataset.cifar100 import load_cifar100
from utils.dataset.domainnet import load_domainnet

class DatasetSplit(Dataset):
    '''
    dataset: refer to all dataset
    indx

    '''
    def __init__(self, dataset, idxs, noise_switch=False,noise_level=1,client_id=0,client_num=1,scaling_factor=0.5):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.noise_switch=noise_switch
        self.mean=0
        if client_id == 0:
            client_id=1
        self.std=noise_level*client_id / client_num * scaling_factor

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        if self.noise_switch is False:
            return tensor(image).float(), tensor(label).float()
        else:
            image=image.numpy()+np.random.normal(self.mean,self.std,image.shape)
            return tensor(image).float(), tensor(label).float()


def load_dataset(dataset_name="mnist",data_base_path=None):
    train_dataset = None
    test_dataset = None
    if data_base_path is None:
        data_base_path = getroot_path() + "/data/"
    if dataset_name == "mnist":
        train_dataset, test_dataset = load_mnist(dataset_name=dataset_name,data_base_path=data_base_path)

    elif dataset_name == "fmnist":
        train_dataset, test_dataset = load_fmnist(dataset_name=dataset_name,data_base_path=data_base_path)

    elif dataset_name == "cifar10":
        train_dataset, test_dataset = load_cifar10(dataset_name=dataset_name,data_base_path=data_base_path)

    elif dataset_name == "cifar100":
        train_dataset, test_dataset = load_cifar100(dataset_name=dataset_name,data_base_path=data_base_path)

    elif dataset_name == "pacs":
        train_dataset,test_dataset=load_pacs()

    elif dataset_name == "domainnet":
        train_dataset,test_dataset=load_domainnet()

    return train_dataset, test_dataset


def get_paratition_data(paratition_type="home",dataset_name="mnist",num_client=1,unbalance_sgm=0.0,dir_alpha=0.5,
                        min_require_sample_size=None,seed=2023):

    train_dataset, test_dataset=load_dataset(dataset_name=dataset_name)

    user_group = paratition_data_index(paratition_type=paratition_type,dataset=train_dataset,num_client=num_client,
                                       unbalance_sgm=unbalance_sgm,dir_alpha=dir_alpha,
                                       min_require_sample_size=min_require_sample_size,seed=seed)

    return train_dataset, test_dataset, user_group

def data_class_num(dataset_name="mnist"):
    class_num=0
    if dataset_name == "mnist":
        class_num=10
    elif dataset_name == "fmnist":
        class_num=10
    elif dataset_name == "emnist":
        class_num = 47
    elif dataset_name == "cifar10":
        class_num = 10
    elif dataset_name == "cifar100":
        class_num = 100
    elif dataset_name == "pacs":
        class_num = 7
    elif dataset_name == "domainnet":
        class_num = 345
    return class_num


if __name__ == '__main__':
    load_dataset(dataset_name="fmnist")