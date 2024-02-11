#-*- coding:utf-8 -*-

import torch.nn as nn

import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../")))
sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class Feature(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

    def forward(self, x):
        x = x.float()
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        return out


class ResNet9(nn.Module):
    def __init__(self, in_channels=None, num_classes=None,dataset_name="mnist"):
        super().__init__()
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

        else:
            self.num_classes = num_classes
            self.in_channels = in_channels
            exit()

        self.feature = Feature(self.in_channels)

        self.classifier = nn.Sequential(nn.AdaptiveMaxPool2d((1, 1)),
                                        nn.Flatten(),
                                        nn.Dropout(0.2),
                                        nn.Linear(512, self.num_classes))

    def forward(self, x):
        x = x.float()
        out= self.feature(x)
        out = self.classifier(out)
        return out

if __name__ == '__main__':

    from utils.data_utils import load_dataset
    from torch.utils.data import DataLoader
    from utils.pytotch_utils import criterion_function, optimizer_function
    import torch
    from utils.evaluate_utils import Evaluate
    import numpy as np

    dataname = "fmnist"
    train_dataset, test_dataset = load_dataset(dataname)
    model=ResNet9(dataset_name=dataname).to("cuda:0")

    criterion = criterion_function('cross')
    optimizer_fun = optimizer_function('sgd')
    optimizer = optimizer_fun(model.parameters(), lr=0.001)

    for epoch in range(200):
        acc_res=[]
        for batch_idx, (x, labels) in enumerate(DataLoader(test_dataset, batch_size=32)):
            model.zero_grad()
            x=x.to("cuda:0")
            labels=labels.to("cuda:0")
            log_probs = model(x)

            loss = criterion(log_probs, labels.long())
            _, predicted = torch.max(log_probs, 1)
            res = Evaluate(labels.cpu().numpy(), predicted.cpu().numpy())
            acc_res.append(res["acc"])
            loss.backward()
            optimizer.step()
        print(np.mean(acc_res))