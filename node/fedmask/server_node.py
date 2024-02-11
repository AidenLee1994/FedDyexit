# -*- coding:utf-8 -*-
# date:2023/6/4 17:06
# Author: 奈斯兔米特油
# WeChat: Mar20th94
# SCUT PhD.Stu

from utils.pytorch_utils import criterion_function,optimizer_function
from core.node_trainer import NodeTrainer
import numpy as np


class Server_Node(NodeTrainer):

    def __init__(self, id=None,model=None,args=None,train_data_loader=None,test_data_loader=None):
        super(Server_Node, self).__init__(id=id, model=model, args=args, train_data_loader=train_data_loader, test_data_loader= test_data_loader)
        self.criterion_fun = criterion_function(args.criterion)
        self.optimizer_fun = optimizer_function(args.optimizer)

    # 聚合方法
    def aggregation(self, model_parameters):
        max_parameters_keys = set()
        for model_parameter in model_parameters:
            keys = model_parameter.keys()
            for key in keys:
                max_parameters_keys.add(key)

        averaged_params = dict()
        for k in max_parameters_keys:
            temp_w = []
            for local_w in model_parameters:
                temp_w.append(local_w[k])
            averaged_params[k] = sum(temp_w) / len(temp_w)
        return averaged_params

    def cal_model_part_params_nums(self):
        total_params = 0
        for name, parameter in self.model.cpu().named_parameters():
            if "weight" in name:
                tmp = parameter.detach().numpy()
                tmp_mask = np.where(np.abs(tmp) >= self.args.fedmask_binarize_threshold, 1, 0)
                total_params += np.sum(tmp_mask)
        return total_params / 1e6