# -*- coding:utf-8 -*-
# date:2023/5/30 17:07
# Author: 奈斯兔米特油
# WeChat: Mar20th94
# SCUT PhD.Stu
from utils.pytorch_utils import criterion_function,optimizer_function
from core.node_trainer import NodeTrainer
from utils.evaluate_utils import Evaluate
import torch
import numpy as np

class Server_Node(NodeTrainer):

    def __init__(self, id=None,model=None,args=None,train_data_loader=None,test_data_loader=None):
        super(Server_Node, self).__init__(id=id, model=model, args=args, train_data_loader=train_data_loader, test_data_loader= test_data_loader)
        self.criterion_fun = criterion_function(args.criterion)
        self.optimizer_fun = optimizer_function(args.optimizer)

    def get_model_part_params(self):
        exitblock_name = "feature"
        part_model_params = self.model._modules[exitblock_name].cpu().state_dict()

        tmp_params = dict()
        for name in part_model_params:
            if "weight" in name:
                tmp = part_model_params[name]
                tmp_params[exitblock_name + "." + name] = tmp

        return tmp_params

    # 设置模型的部分参数
    def set_model_part_params(self, aggregation_model):
        all_model_params = self.model.cpu().state_dict()
        for layer_name in aggregation_model.keys():
            all_model_params[layer_name] = aggregation_model[layer_name]

        self.model.load_state_dict(all_model_params)

    # 聚合方法
    def aggregation(self, model_parameters):
        averaged_params = model_parameters[0]
        for k in averaged_params.keys():
            temp_w = []
            for local_w in model_parameters:
                temp_w.append(local_w[k])
            averaged_params[k] = sum(temp_w) / len(temp_w)
        return averaged_params

    def cal_model_part_params_nums(self):
        total_params = 0

        exitblock_name = "feature"
        part_model_params = self.model._modules[exitblock_name].cpu().state_dict()

        for name in part_model_params:
            if "weight" in name:
                param = part_model_params[name].numel()
                total_params += param
        return total_params / 1e6



