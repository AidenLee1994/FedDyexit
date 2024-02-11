# -*- coding:utf-8 -*-
__author__ = 'Aiden Lee'
__date__ = '2022/4/28 15:48'

import torch
from torch import nn


class EE_net(nn.Module):
    def __init__(self,exit_threshold=0.05):
        super(EE_net, self).__init__()
        self.train_mode=0

        self.exit_threshold=torch.tensor([exit_threshold],dtype=torch.float)

        self.backbone=nn.ModuleList()
        self.exitblock=nn.ModuleList()

        self.train_exit_index={}
        self.test_exit_index={}
        self.exit_loss_weights=None

    #设置主干网络

    def set_exit_threshold(self,threshold):
        self.exit_threshold = torch.tensor([threshold], dtype=torch.float)

    def build_backbone(self):
        pass

    #设置退出网络

    def build_exitblock(self):
        pass

    #设置退出的标准
    def exit_criterion(self,x):
        with torch.no_grad():
            res_softmax=nn.functional.softmax(x,dim=-1)
            entr=-torch.sum(res_softmax*torch.log(res_softmax)).cpu()
            if entr<self.exit_threshold:
                return True
            else:
                return False

    def set_train_mode(self,mode=0):
        self.train_mode=mode

    def set_exit_threshold(self,exit_threshold):
        self.exit_threshold = torch.tensor([exit_threshold], dtype=torch.float)


    #设置是否快速推理
    def set_test_mode(self,mode=True):
        if mode:
            self.eval()
        self.test_mode=mode

    def forward_training_backborn(self,x):
        for back in self.backbone:
            x=back(x)
        cal_res=self.exitblock[-1](x)
        return cal_res

    def forward_training_exits(self,x):
        cal_res=[]
        for back in self.backbone.parameters():
            back.requires_grad=False

        for index, (back,exit) in enumerate(zip(self.backbone,self.exitblock)):
            x=back(x)
            res_exit=exit(x)

            if self.exit_criterion(res_exit):
                cal_res.append(res_exit)
                break
            else:
                cal_res.append(res_exit)
        return cal_res

    def forward_training_joint(self,x):
        cal_res=[]
        for index,(back,exit) in enumerate(zip(self.backbone,self.exitblock)):

            x=back(x)

            res_exit=exit(x)

            if self.exit_criterion(res_exit):

                cal_res.append(res_exit)
                if index in self.train_exit_index.keys():
                    self.train_exit_index[index]+=1
                else:
                    self.train_exit_index[index]=1
                break
            else:
                if index in self.train_exit_index.keys():
                    self.train_exit_index[index]+=1
                else:
                    self.train_exit_index[index]=1
                cal_res.append(res_exit)
        return cal_res

    def forward_testing_backborn(self,x):
        with torch.no_grad():
            for back in self.backbone:
                x=back(x)
            cal_res=self.exitblock[-1](x)
            return cal_res


    def forward_testing_exit(self,x):
        with torch.no_grad():
            for index, (back, exit) in enumerate(zip(self.backbone, self.exitblock)):
                x = back(x)
                res_exit = exit(x)
                if self.exit_criterion(res_exit):
                    if index in self.test_exit_index.keys():
                        self.test_exit_index[index] += 1
                    else:
                        self.test_exit_index[index] = 1
                    break
                else:
                    if index in self.test_exit_index.keys():
                        self.test_exit_index[index] += 1
                    else:
                        self.test_exit_index[index] = 1

            return res_exit

    def forward(self,x):
        if self.train_mode == 0:
            res=self.forward_training_backborn(x)
        elif self.train_mode == 1:
            res = self.forward_testing_backborn(x)
        elif self.train_mode == 2:
            res=self.forward_training_joint(x)
        elif self.train_mode == 3:
            res = self.forward_testing_exit(x)

        return res
