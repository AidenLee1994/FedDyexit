# -*- coding:utf-8 -*-
# date:2023/5/30 15:45
# Author: 奈斯兔米特油
# WeChat: Mar20th94
# SCUT PhD.Stu
from core.node_trainer import NodeTrainer
from utils.pytorch_utils import criterion_function,optimizer_function
from utils.evaluate_utils import Evaluate
import torch
import numpy as np
import copy
from utils.model_utils import load_model

def MMD(x, y, kernel, device='cpu'):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2. * XY)


class Client_Node(NodeTrainer):

    def __init__(self, id=None, model=None, args=None, train_data_loader=None, test_data_loader=None):
        super(Client_Node, self).__init__(id=id, model=model, args=args, train_data_loader=train_data_loader,
                                          test_data_loader=test_data_loader)
        self.criterion_fun = criterion_function(args.criterion)
        self.optimizer_fun = optimizer_function(args.optimizer)

        self.mu = self.args.fedphp_mu / self.args.comm_round
        self.lamda = self.args.fedphp_lamda

        self.model_s=copy.deepcopy(self.model)
        for param in self.model_s.parameters():
            param.requires_grad = False

    def set_model_part_params(self, model_parameters):
        mu=self.mu*self.id

        for new_param, old_param in zip(model_parameters.parameters(), self.model_s.parameters()):
            old_param.data = new_param.data.clone()

        for new_param, old_param in zip(model_parameters.parameters(), self.model.parameters()):
            old_param.data = new_param * (1 - mu) + old_param * mu


    def node_train(self, data=None):

        # if self.cuda_count >0:
        #     self.model=torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        self.model_s.to(self.device)
        self.model.train()

        if data is None:
            train_data_loader = self.train_data_loader
        else:
            train_data_loader = data

        criterion = self.criterion_fun

        optimizer = self.optimizer_fun(self.model.parameters(), lr=self.args.lr)

        result = {"acc": [], "pre": [], "f1": [], "recall": [], "d_idx": [], "loss": []}
        for epoch in range(self.args.epochs):
            for batch_idx, (x, labels) in enumerate(train_data_loader):
                x, labels = x.to(self.device), labels.to(self.device)
                self.model.zero_grad()

                log_probs = self.model(x)

                loss = criterion(log_probs, labels.long()) * (1 - self.lamda)
                loss += MMD(self.model(x), self.model_s(x), 'rbf', self.device) * self.lamda

                _, predicted = torch.max(log_probs, 1)

                result["loss"].append(loss.item())
                res = Evaluate(labels.cpu().numpy(), predicted.cpu().numpy())
                result["acc"].append(res["acc"])
                result["pre"].append(res["pre"])
                result["f1"].append(res["f1"])
                result["recall"].append(res["recall"])
                result["d_idx"].append(res["d_idx"])

                loss.backward()
                optimizer.step()

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