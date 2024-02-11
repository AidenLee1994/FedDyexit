# -*- coding:utf-8 -*-
# date:2023/6/14 20:02
# Author: 奈斯兔米特油
# WeChat: Mar20th94
# SCUT PhD.Stu
import torch.nn as nn
import math
import torch
import numpy as np
import sys,os
sys.path.append(os.path.abspath(os.path.join(__file__, "../")))
sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))
from model.base_ee import EE_net
from torch.utils.data import Dataset
from torch import tensor

class Lenet(EE_net):
    def __init__(self,dataset_name="mnist",cfg=None):
        super(Lenet, self).__init__()
        self.exit_loss_weights = [0.5, 0.5]
        self.dataset_name=dataset_name

        if cfg is None:
            self.cfg=[80]
        else:
            self.cfg = cfg  # 模型配置

        if self.dataset_name == "mnist":
            self.num_classes = 10
            self.dimention = 2
            self.in_channels=1

        elif self.dataset_name == "fmnist":
            self.num_classes = 10
            self.dimention = 2
            self.in_channels=1

        self.build_backbone()
        self.build_exitblock()
        self.initialize_weights()

    def set_exit_loss_weights(self,weight=None):
        if weight == None:
            self.exit_loss_weights = [0.5, 0.5]
        else:
            self.exit_loss_weights = [weight, 1-weight]
    def build_backbone(self):

        b1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 10, kernel_size=3, padding=1, bias= False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )
        self.backbone.append(b1)

        global b2

        if self.dataset_name == "mnist":
            b2=nn.Sequential(
                nn.Conv2d(10, 20, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(20 * 8 * 8, self.cfg[-1]),
            )
        elif self.dataset_name == "fmnist":
            b2 = nn.Sequential(
                nn.Conv2d(10, 20, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Linear(20 * 7 * 7, self.cfg[-1]),
            )

        self.backbone.append(b2)

    def build_exitblock(self):

        ee=[nn.Conv2d(10,3,kernel_size=3,padding=1,bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),nn.Flatten()]

        if self.dataset_name == "mnist":
            ee.append(nn.Linear(3*8*8,self.num_classes))
        elif self.dataset_name == "fmnist":
            ee.append(nn.Linear(3*7*7,self.num_classes))
        self.exitblock.append(nn.Sequential(*ee))

        eeF=nn.Sequential(nn.Linear(self.cfg[-1],self.num_classes))
        self.exitblock.append(eeF)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()




if __name__ == '__main__':

    class DatasetSplit(Dataset):
        '''
        dataset: refer to all dataset
        indx

        '''

        def __init__(self, dataset, idxs, noise_switch=False, noise_level=1, client_id=0, client_num=1,
                     scaling_factor=0.5):
            self.dataset = dataset
            self.idxs = [int(i) for i in idxs]
            self.noise_switch = noise_switch
            self.mean = 0
            if client_id == 0:
                client_id = 1
            self.std = noise_level * client_id / client_num * scaling_factor

        def __len__(self):
            return len(self.idxs)

        def __getitem__(self, item):
            image, label = self.dataset[self.idxs[item]]
            if self.noise_switch is False:
                return tensor(image), tensor(label)
            else:
                image = image.numpy() + np.random.normal(self.mean, self.std, image.shape)
                return tensor(image), tensor(label)


    from torch.utils.data import DataLoader
    from utils.data_utils import load_dataset
    dataname = "fmnist"
    model = Lenet(dataset_name=dataname)

    model.set_train_mode(2)

    train_dataset, test_dataset = load_dataset(dataname)

    a=DatasetSplit(test_dataset,[1,2,3,4,5], noise_switch=True,noise_level=1,client_id=1,client_num=1,scaling_factor=0.5)

    dataloader=DataLoader(a,batch_size=16,shuffle=True)
    for batch_idx, (data,target) in enumerate(dataloader):
        print(model(data.float()))