# -*- coding:utf-8 -*-
# date:2023/5/30 14:27
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

        self.mu=self.args.ditto_mu

        self.criterion_fun = criterion_function(args.criterion)
        self.optimizer_fun = optimizer_function(args.optimizer)

        self.model_per=copy.deepcopy(self.model)


    def pre_node_train(self,data=None):
        self.model_per.to(self.device)
        self.model_per.train()

        if data is None:
            train_data_loader = self.train_data_loader
        else:
            train_data_loader = data

        criterion = self.criterion_fun

        optimizer = PerturbedGradientDescent(
            self.model_per.parameters(), lr=self.args.lr, mu=self.mu)

        for epoch in range(self.args.ditto_pre_epochs):
            for batch_idx, (x, labels) in enumerate(train_data_loader):
                x, labels = x.to(self.device), labels.to(self.device)
                self.model_per.zero_grad()

                log_probs = self.model_per(x)

                loss = criterion(log_probs, labels.long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step(self.model.parameters(),self.device)


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

        result = {"acc": [], "pre": [], "f1": [], "recall": [], "d_idx": [],  "loss": []}
        for epoch in range(self.args.epochs):
            for batch_idx, (x, labels) in enumerate(train_data_loader):
                x, labels = x.to(self.device), labels.to(self.device)
                self.model.zero_grad()

                log_probs = self.model(x)

                loss = criterion(log_probs, labels.long())
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

        result["acc"]=np.mean(result["acc"])
        result["pre"]=np.mean(result["pre"])
        result["f1"]=np.mean(result["f1"])
        result["recall"]=np.mean(result["recall"])
        result["d_idx"]=np.mean(result["d_idx"])
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

        result = {"acc": [], "pre": [], "f1": [], "recall": [], "d_idx": [],  "loss": []}
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

    class torch_Dataset(Dataset):
        def __init__(self, data_name, file_type, file_base_path="../data/"):
            self.data_name = data_name
            self.file_type = file_type
            self.file_path = file_base_path + data_name + "/" + self.data_name + "_" + self.file_type + ".csv"
            self.data = pd.read_csv(self.file_path)
            self.x, self.targets = self.del_data()

        def del_data(self):
            data_x = []
            data_y = []
            for index, row in self.data.iterrows():
                data_x.append([eval(row["x"])])
                data_y.append(row["y"])
            return np.array(data_x), np.array(data_y)

        def __getitem__(self, item):
            return self.x[item], self.targets[item]

        def __len__(self):
            return len(self.targets)

    import argparse

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

    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader


    train_dataset = torch_Dataset("kolod", "test", "../../data/")
    train_dataloader = DataLoader(train_dataset, batch_size=64)

    from model.lenet import Lenet

    a = Client_Node(args=parser.parse_args())

    model = Lenet(dataset_name="kolod")

    a.set_model(model)

    a.node_train(train_dataloader)