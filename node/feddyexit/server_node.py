# -*- coding:utf-8 -*-
# date:2023/6/15 10:40
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


    def get_model_part_params(self):

        part_model_params = self.model.cpu().state_dict()
        tmp_params = dict()
        for name in part_model_params:
            if "weight" in name:
                tmp=part_model_params[name]
                tmp_params[name] = tmp
        return tmp_params

    # 设置模型的部分参数
    def set_model_part_params(self, model_parameters):
        all_model_params = self.model.cpu().state_dict()
        for layer_name in model_parameters.keys():
            all_model_params[layer_name]=model_parameters[layer_name]
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