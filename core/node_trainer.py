# -*- coding:utf-8 -*-
# date: 2023/5/22 下午4:47
# -*- coding:utf-8 -*-
# date:2022/11/22 15:37
from abc import ABC
import torch
from collections import OrderedDict


class NodeTrainer(ABC):

    def __init__(self, id=None, model=None, args=None, train_data_loader=None, test_data_loader=None):
        self.id = id  # server编号默认为0,其余客户端编号从1开始
        self.model = model
        self.args = args
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.device = None
        self.gpu_switch=args.gpu_switch
        self.gpu_id=args.gpu_id
        self.Device()  # 检测设备是否具有cuda可以使用

    # 检测使用设备是否存在，以及调整使用设备
    def Device(self):
        # 判定是否需要启动显卡
        if torch.cuda.is_available() is False:
            self.device = "cpu"
        else:
            if self.gpu_switch is False:
                self.device = "cpu"
            else:
                cuda_count = torch.cuda.device_count()
                if self.gpu_id<cuda_count:
                    self.device = "cuda:"+str(self.gpu_id)
                else:
                    self.device = "cuda:0"

    # 设置模型所属客户端
    def set_id(self, id):
        self.id = id

    # 获取模型所属客户端编号
    def get_id(self):
        return self.id

    # 设置模型
    def set_model(self, model):
        self.model = model

    # 获取模型
    def get_model(self):
        return self.model

    # 获取模型的所有参数
    def get_model_all_params(self):
        return self.model.cpu().state_dict()

    # 设置模型的所有参数
    def set_model_all_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    # 设置训练数据
    def set_train_data_loader(self, train_data_loader):
        self.train_data_loader = train_data_loader

    # 设置测试数据
    def set_test_data_loader(self, test_data_loader):
        self.test_data_loader = test_data_loader

    # 读取训练数据
    def get_train_data_loader(self):
        return self.train_data_loader

    # 读取测试数据
    def get_test_data_loader(self):
        return self.test_data_loader

    # 获取模型的部分参数
    def get_model_part_params(self):
        pass

    # 设置模型的部分参数
    def set_model_part_params(self, model_parameters):
        pass

    # 计算整个模型参数大小
    def cal_model_all_params_nums(self, model):
        total = sum([param.nelement() if param.requires_grad else 0 for param in model.parameters()])
        return total / 1e6

    # 计算部分模型参数大小
    def cal_model_part_params_nums(self):
        pass

    #training
    def pre_node_train(self):
        pass
    def node_train(self):
        pass

    def post_node_train(self):
        pass

    #testing
    def pre_node_test(self):
        pass

    def node_test(self):
        pass

    def post_node_test(self):
        pass




