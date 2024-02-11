# -*- coding:utf-8 -*-
# date:2023/6/4 17:06
# Author: 奈斯兔米特油
# WeChat: Mar20th94
# SCUT PhD.Stu
import copy

from core.node_trainer import NodeTrainer
from utils.pytorch_utils import criterion_function,optimizer_function
from utils.evaluate_utils import Evaluate
import torch
import numpy as np
from utils.model_utils import load_model

class Client_Node(NodeTrainer):

    def __init__(self, id=None, model=None, args=None, train_data_loader=None, test_data_loader=None):
        super(Client_Node, self).__init__(id=id, model=model, args=args, train_data_loader=train_data_loader,
                                          test_data_loader=test_data_loader)
        self.criterion_fun = criterion_function(args.criterion)
        self.optimizer_fun = optimizer_function(args.optimizer)
        self.binarize_mask = dict()

    def get_model_part_params(self):
        part_model_params = self.model.cpu().state_dict()
        tmp_params_mask = dict()
        for name in part_model_params:
            if "weight" in name:
                tmp = part_model_params[name].numpy()
                tmp_mask = np.where(np.abs(tmp) >= self.args.fedmask_binarize_threshold, 1, 0)
                tmp_params_mask[name] = tmp_mask
                self.binarize_mask[name] = tmp_mask
        return tmp_params_mask

    def cal_model_part_params_nums(self):
        total_params = 0
        model=copy.deepcopy(self.model)
        for name, parameter in model.cpu().named_parameters():
            if "weight" in name:
                tmp = parameter.detach().numpy()
                tmp_mask = np.where(np.abs(tmp) >= self.args.fedmask_binarize_threshold, 1, 0)
                total_params += np.sum(tmp_mask)
        return total_params / 1e6

    # 设置模型的部分参数
    def set_model_part_params(self, aggregation_model_mask):
        all_model_params = self.model.cpu().state_dict()
        for layer_name in self.binarize_mask.keys():
            mask = aggregation_model_mask[layer_name] * self.binarize_mask[layer_name]

            layer = all_model_params[layer_name].numpy()

            all_model_params[layer_name] = torch.from_numpy(layer * mask)

        self.model.load_state_dict(all_model_params)


    def node_train(self, data=None):

        # if self.cuda_count >0:
        #     self.model=torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        self.model.train()

        if data is None:
            train_data_loader = self.train_data_loader
        else:
            train_data_loader = data

        criterion = self.criterion_fun

        optimizer = self.optimizer_fun(self.model.parameters(), lr=self.args.lr,momentum=self.args.momentum)


        for epoch in range(self.args.epochs):
            for batch_idx, (x, labels) in enumerate(train_data_loader):
                x, labels = x.to(self.device), labels.to(self.device)
                self.model.zero_grad()

                log_probs = self.model(x)

                loss = criterion(log_probs, labels.long())

                loss.backward()
                optimizer.step()

    def node_test(self, data=None):

        # if self.cuda_count > 0:
        #     self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        self.model.eval()

        send_params_num = self.cal_model_part_params_nums()

        if data is None:
            test_data_loader = self.test_data_loader
        else:
            test_data_loader = data

        result = {"acc": [], "pre": [], "f1": [], "recall": [], "d_idx": [], "loss": []}
        criterion = self.criterion_fun

        with torch.no_grad():
            for batch_idx, (x, labels) in enumerate(test_data_loader):
                x, labels = x.to(self.device), labels.to(self.device)
                self.model.zero_grad()
                log_probs = self.model(x)
                loss = criterion(log_probs, labels.long())
                result["loss"].append(loss.item())
                _, predicted = torch.max(log_probs, 1)
                res = Evaluate(labels.cpu().numpy(), predicted.cpu().numpy())
                result["acc"].append(res["acc"])
                result["pre"].append(res["pre"])
                result["f1"].append(res["f1"])
                result["recall"].append(res["recall"])
                result["d_idx"].append(res["d_idx"])

        result["acc"] = np.mean(result["acc"])
        result["pre"] = np.mean(result["pre"])
        result["f1"] = np.mean(result["f1"])
        result["recall"] = np.mean(result["recall"])
        result["d_idx"] = np.mean(result["d_idx"])
        result["loss"] = np.mean(result["loss"])
        result["send_params_num"] = send_params_num
        return result

