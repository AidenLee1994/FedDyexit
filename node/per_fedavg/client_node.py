# -*- coding:utf-8 -*-
# date:2023/5/30 13:59
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


    def node_train(self, data=None):

        self.model.to(self.device)
        self.model.train()

        if data is None:
            train_data_loader = self.train_data_loader
        else:
            train_data_loader = data

        criterion = self.criterion_fun


        epoch_loss = []
        flag=True
        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data_loader):
                labels = labels.type(torch.LongTensor)
                x, labels = x.to(self.device), labels.to(self.device)
                origin_model = copy.deepcopy(self.model)
                if flag==True:
                    grads1=self.compute_grad(x,labels,origin_model)

                    per_fedavg_alpha=torch.tensor(self.args.per_fedavg_alpha).to(self.device)
                    for param, grad in zip(origin_model.parameters(), grads1):
                        if grad is None:
                            continue
                        param.data.sub_(per_fedavg_alpha * grad)
                    flag=False
                else:

                    grads2 = self.compute_grad(x, labels, origin_model)

                    per_fedavg_beta = torch.tensor(self.args.per_fedavg_beta).to(self.device)
                    for param, grad in zip(self.model.parameters(), grads2):
                        if grad is None:
                            continue
                        param.data.sub_(per_fedavg_beta * grad)
                    flag=True

            for batch_idx, (x,labels) in enumerate(train_data_loader):
                labels = labels.type(torch.LongTensor)
                x, labels = x.to(self.device), labels.to(self.device)
                log_probs = self.model(x)
                loss = criterion(log_probs, labels)

                batch_loss.append(loss.item())

            len_batch_loss=len(batch_loss)
            if len_batch_loss == 0:
                epoch_loss.append(0)
            else:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))

    def compute_grad(self, x, labels, model):
        x, labels = x.to(self.device), labels.to(self.device)
        model = model.to(self.device)
        model.train()
        model.zero_grad()
        log_probs = model(x)
        criterion = self.criterion_fun
        loss = criterion(log_probs, labels)
        grads = torch.autograd.grad(loss, model.parameters(),allow_unused=True,retain_graph=True)
        return grads


    def node_test(self, data=None):
        self.model.to(self.device)
        self.model.eval()

        new_model = load_model(model_name=self.args.model_name, dataset_name=self.args.dataset_name)
        send_params_num = self.cal_model_all_params_nums(new_model)

        if data is None:
            test_data_loader = self.test_data_loader
        else:
            test_data_loader = data

        result = {"acc": [], "pre": [], "f1": [], "recall": [], "d_idx": [], "loss": []}
        criterion = self.criterion_fun

        with torch.no_grad():
            for batch_idx, (x, labels) in enumerate(test_data_loader):
                labels = labels.type(torch.LongTensor)
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