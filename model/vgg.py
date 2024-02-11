# -*- coding:utf-8 -*-
# date:2022/10/25 23:36
import torch.nn as nn
import math
import torch
import numpy as np
import sys,os
sys.path.append(os.path.abspath(os.path.join(__file__, "...")))
sys.path.append(os.path.abspath(os.path.join(__file__, ".../...")))


class VGG(nn.Module):

    def __init__(self,dataset_name="mnist",cfg=None,num_classes=None,in_channels=None,dimention=None):
        super(VGG,self).__init__()

        if cfg is None:
            self.cfg=[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,'M', 512, 512, 512]
        else:
            self.cfg = cfg  # 模型配置

        if dataset_name == "mnist":
            self.num_classes = 10
            self.dimention = 2
            self.in_channels=1

        elif dataset_name == "fmnist":
            self.num_classes = 10
            self.dimention = 2
            self.in_channels = 1


        elif dataset_name == "emnist":

            self.num_classes = 47
            self.dimention = 2
            self.in_channels = 1

        else:
            self.num_classes = num_classes  # 数据类型
            self.in_channels = in_channels  # 输入维度
            self.dimention = dimention  # 数据维度（一维或二维）

        self.feature= self.make_layers(self.cfg,batch_norm=True)

        self.classifier=self.make_classifer_layers(self.cfg,dataset_name)

        self.initialize_weights()

    def make_classifer_layers(self,cfg,dataset_name):
        layers=[]
        if dataset_name == "mnist":
            layers+=[nn.Linear(cfg[-1] * 6 * 6, 256),nn.LeakyReLU(inplace=True),nn.Dropout(0.5)]
        elif dataset_name == "fmnist":
            layers += [nn.Linear(cfg[-1] * 6 * 6, 256), nn.LeakyReLU(inplace=True), nn.Dropout(0.5)]
        elif dataset_name == "emnist":
            layers += [nn.Linear(cfg[-1] * 6 * 6, 256), nn.LeakyReLU(inplace=True),nn.Dropout(0.5)]

        layers+=[nn.Linear(256, self.num_classes)]
        return nn.Sequential(*layers)


    def make_layers(self,cfg,batch_norm=False):
        layers = []
        in_channels = self.in_channels
        for v in cfg:
            if v == 'M':
                if self.dimention == 2:
                    layers += [nn.MaxPool2d(kernel_size=3, stride=2)]
                elif self.dimention == 1:
                    layers += [nn.MaxPool1d(kernel_size=3, stride=2)]
            else:
                if self.dimention == 2:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=2, padding=1, bias=False)
                    if batch_norm:
                        layers += [conv2d, nn.BatchNorm2d(v), nn.LeakyReLU(inplace=True)]
                    else:
                        layers += [conv2d, nn.LeakyReLU(inplace=True)]
                elif self.dimention == 1:
                    conv1d = nn.Conv1d(in_channels, v, kernel_size=2, padding=1, bias=False)
                    if batch_norm:
                        layers += [conv1d, nn.BatchNorm1d(v), nn.LeakyReLU(inplace=True)]
                    else:
                        layers += [conv1d, nn.LeakyReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self,x):
        x=x.float()
        x=self.feature(x)

        x=x.view(x.size(0),-1)
        x=self.classifier(x)
        return x

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
    from torch.utils.data import DataLoader
    from utils.data_utils import load_dataset
    dataname = "mnist"
    train_dataset, test_dataset = load_dataset(dataname)
    dataloader=DataLoader(test_dataset,batch_size=16,shuffle=True)
    for batch_idx, (data,target) in enumerate(dataloader):
        model = VGG(dataset_name=dataname)
        pre=model(data)
        print(np.shape(pre))
        print(target)
        exit()