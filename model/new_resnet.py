# -*- coding:utf-8 -*-
# date:2023/6/14 20:02
# Author: 奈斯兔米特油
# WeChat: Mar20th94
# SCUT PhD.Stu
import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../")))
sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))
import torch.nn as nn
import math
import numpy as np
from torch.utils.data import Dataset
from torch import tensor
from model.base_ee import EE_net


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: 
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class backbone1(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.in_channels=in_channels
            
        self.conv1 = conv_block(self.in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
    
    def forward(self, x):
        x = x.float()
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        return out    
    
class backbone2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
    
    def forward(self,x):
        out = self.conv3(x)
        out = self.conv4(out)
        out = self.res2(out) + out
        return out


class Resnet(EE_net):
    def __init__(self,dataset_name="mnist"):
        super(Resnet, self).__init__()
        self.exit_loss_weights = [0.5, 0.5]
        self.dataset_name=dataset_name

        if dataset_name == "mnist":
            self.num_classes = 10
            self.in_channels=1

        elif dataset_name == "fmnist":
            self.num_classes = 10
            self.in_channels=1

        elif dataset_name == "cifar10":
            self.num_classes = 10
            self.in_channels = 3

        elif dataset_name == "cifar100":
            self.num_classes = 100
            self.in_channels = 3

        elif dataset_name == "pacs":
            self.num_classes = 7
            self.in_channels = 3

        elif dataset_name == "domainnet":
            self.num_classes = 345
            self.in_channels = 3

        self.build_backbone()
        self.build_exitblock()
        self.initialize_weights()

    def set_exit_loss_weights(self,weight=None):
        if weight is None:
            self.exit_loss_weights = [0.5, 0.5]
        else:
            self.exit_loss_weights = [weight, 1-weight]
    def build_backbone(self):

        self.backbone.append(backbone1(self.in_channels))

        self.backbone.append(backbone2())

    def build_exitblock(self):

        ee=[conv_block(128, 64, pool=True),
                            nn.Conv2d(64,32,kernel_size=3,padding=1),
                            nn.MaxPool2d(kernel_size=2,stride=2),
                            nn.ReLU(inplace=True),
                            nn.Flatten(),
            ]

        if self.dataset_name == "mnist":
             ee.append(nn.Linear(512,self.num_classes))

        elif self.dataset_name == "fmnist":
             ee.append(nn.Linear(228,self.num_classes))

        elif self.dataset_name == "cifar10":
             ee.append(nn.Linear(512,self.num_classes))

        elif self.dataset_name == "cifar100":
             ee.append(nn.Linear(512,self.num_classes))

        elif self.dataset_name == "pacs":
             ee.append(nn.Linear(512,self.num_classes))

        elif self.dataset_name == "domainnet":
             ee.append(nn.Linear(512,self.num_classes))

        self.exitblock.append(nn.Sequential(*ee))

        eeF=nn.Sequential(nn.AdaptiveMaxPool2d((1, 1)),
                                        nn.Flatten(),
                                        nn.Dropout(0.2),
                                        nn.Linear(512, self.num_classes))
        
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
    model = Resnet(dataset_name=dataname)

    model.set_train_mode(2)

    train_dataset, test_dataset = load_dataset(dataname)

    a=DatasetSplit(test_dataset,[1,2,3,4,5], noise_switch=True,noise_level=1,client_id=1,client_num=1,scaling_factor=0.5)

    dataloader=DataLoader(a,batch_size=16,shuffle=True)
    for batch_idx, (data,target) in enumerate(dataloader):
        print(model(data.float()))