# -*- coding:utf-8 -*-
# date:2023/5/30 14:28
# Author: 奈斯兔米特油
# WeChat: Mar20th94
# SCUT PhD.Stu

from utils.pytorch_utils import criterion_function,optimizer_function
from core.node_trainer import NodeTrainer


class Server_Node(NodeTrainer):

    def __init__(self, id=None,model=None,args=None,train_data_loader=None,test_data_loader=None):
        super(Server_Node, self).__init__(id=id, model=model, args=args, train_data_loader=train_data_loader, test_data_loader= test_data_loader)
        self.criterion_fun = criterion_function(args.criterion)
        self.optimizer_fun = optimizer_function(args.optimizer)


    # 聚合方法
    def aggregation(self, model_parameters):
        averaged_params = model_parameters[0]
        for k in averaged_params.keys():
            temp_w = []
            for local_w in model_parameters:
                temp_w.append(local_w[k])
            averaged_params[k] = sum(temp_w) / len(temp_w)
        return averaged_params