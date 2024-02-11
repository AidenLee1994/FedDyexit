# -*- coding:utf-8 -*-
# date:2023/6/1 14:44
# Author: 奈斯兔米特油
# WeChat: Mar20th94
# SCUT PhD.Stu
from collections import defaultdict
from core.node_trainer import NodeTrainer
from utils.pytorch_utils import criterion_function,optimizer_function
from utils.evaluate_utils import Evaluate
import torch
import numpy as np
from torch import nn
from utils.model_utils import load_model
import copy

class Client_Node(NodeTrainer):

    def __init__(self, id=None, model=None, args=None, train_data_loader=None, test_data_loader=None):
        super(Client_Node, self).__init__(id=id, model=model, args=args, train_data_loader=train_data_loader,
                                          test_data_loader=test_data_loader)
        self.criterion_fun = criterion_function(args.criterion)
        self.optimizer_fun = optimizer_function(args.optimizer)
        self.logits = None
        self.global_logits = None
        self.loss_mse = nn.MSELoss()

        self.lamda = self.args.feddistill_lamda

    def cal_model_part_params_nums(self):
        import sys
        return sys.getsizeof(self.logits)/1024/1024


    def set_logits(self, global_logits):
        self.global_logits = copy.deepcopy(global_logits)

    def get_logits(self):
        return self.logits

    def agg_func(self,logits):
        """
        Returns the average of the weights.
        """

        for [label, logit_list] in logits.items():
            if len(logit_list) > 1:
                logit = 0 * logit_list[0].data
                for i in logit_list:
                    logit += i.data
                logits[label] = logit / len(logit_list)
            else:
                logits[label] = logit_list[0]

        return logits

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

        optimizer = self.optimizer_fun(self.model.parameters(), lr=self.args.lr)

        logits = defaultdict(list)

        result = {"acc": [], "pre": [], "f1": [], "recall": [], "d_idx": [], "loss": []}
        for epoch in range(self.args.epochs):
            for batch_idx, (x, labels) in enumerate(train_data_loader):
                x, labels = x.to(self.device), labels.to(self.device)
                self.model.zero_grad()

                log_probs = self.model(x)

                loss = criterion(log_probs, labels.long())
                _, predicted = torch.max(log_probs, 1)

                if self.global_logits != None:
                    logit_new = copy.deepcopy(log_probs.detach())
                    for i, yy in enumerate(labels):
                        y_c = yy.item()
                        if type(self.global_logits[y_c]) != type([]):
                            logit_new[i, :] = self.global_logits[y_c].data
                    loss += self.loss_mse(logit_new, log_probs) * self.lamda

                result["loss"].append(loss.item())
                res = Evaluate(labels.cpu().numpy(), predicted.cpu().numpy())
                result["acc"].append(res["acc"])
                result["pre"].append(res["pre"])
                result["f1"].append(res["f1"])
                result["recall"].append(res["recall"])
                result["d_idx"].append(res["d_idx"])


                loss.backward()
                optimizer.step()

        self.logits = self.agg_func(logits)

        result["acc"] = np.mean(result["acc"])
        result["pre"] = np.mean(result["pre"])
        result["f1"] = np.mean(result["f1"])
        result["recall"] = np.mean(result["recall"])
        result["d_idx"] = np.mean(result["d_idx"])
        result["loss"] = np.mean(result["loss"])
        return result

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



if __name__ == '__main__':

    import argparse
    from torch.utils.data import DataLoader
    from utils.data_utils import load_dataset



    parser = argparse.ArgumentParser()

    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                            of optimizer")
    parser.add_argument('--criterion', type=str, default='cross', help="type \
                                of optimizer")
    parser.add_argument('--epochs', type=int, default='10', help="type \
                                    of optimizer")
    parser.add_argument('--lr', type=float, default='0.1', help="type \
                                        of learning rate")
    parser.add_argument('--device_type', type=str, default="gpu",
                        help='gpu or cpu')

    dataname = "emnist"
    model=load_model("lenet",dataname)

    train_dataset, test_dataset = load_dataset(dataname)
    dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    a = Client_Node(args=parser.parse_args())

    a.set_model(model)

    a.node_train(dataloader)