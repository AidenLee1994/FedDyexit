# -*- coding:utf-8 -*-
# date:2023/6/4 15:30
# Author: 奈斯兔米特油
# WeChat: Mar20th94
# SCUT PhD.Stu
from core.node_trainer import NodeTrainer
from utils.pytorch_utils import criterion_function,optimizer_function
from utils.evaluate_utils import Evaluate
import torch
import numpy as np
from utils.model_utils import load_model
from torch.optim import Optimizer
import copy


class PerturbedGradientDescent(Optimizer):
    def __init__(self, params, lr=0.01, mu=0.0):
        default = dict(lr=lr, mu=mu)
        super().__init__(params, default)

    @torch.no_grad()
    def step(self, global_params, device):
        for group in self.param_groups:
            for p, g in zip(group['params'], global_params):
                g = g.to(device)
                d_p = p.grad.data + group['mu'] * (p.data - g.data)
                p.data.add_(d_p, alpha=-group['lr'])


class Client_Node(NodeTrainer):

    def __init__(self, id=None, model=None, args=None, train_data_loader=None, test_data_loader=None):
        super(Client_Node, self).__init__(id=id, model=model, args=args, train_data_loader=train_data_loader,
                                          test_data_loader=test_data_loader)
        self.criterion_fun = criterion_function(args.criterion)
        self.optimizer_fun = optimizer_function(args.optimizer)

        self.mu = self.args.fedprox_mu
        self.global_params = copy.deepcopy(list(self.model.parameters()))
        self.optimizer = PerturbedGradientDescent(
            self.model.parameters(), lr=self.args.lr, mu=self.mu)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=self.args.fedprox_learning_rate_decay_gamma)

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

        # optimizer = self.optimizer_fun(self.model.parameters(), lr=self.args.lr)


        for epoch in range(self.args.epochs):
            for batch_idx, (x, labels) in enumerate(train_data_loader):
                x, labels = x.to(self.device), labels.to(self.device)
                log_probs = self.model(x)

                loss = criterion(log_probs, labels.long())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step(self.global_params, self.device)


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