# -*- coding:utf-8 -*-
# date:2023/6/15 10:40
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

    def init_binarize_mask(self):
        # exitblock_name = self.args.exitblock_name
        # part_model_params = self.model._modules[exitblock_name].cpu().state_dict()

        part_model_params = self.model.cpu().state_dict()

        for name in part_model_params:
            if "weight" in name:
                tmp = part_model_params[name].numpy()
                tmp_mask = np.ones_like(tmp)
                # self.binarize_mask[exitblock_name + "." + name] = tmp_mask
                self.binarize_mask[name] = tmp_mask

    def get_model_part_params(self):
        #exitblock_name = self.args.exitblock_name
        # part_model_params = self.model._modules[exitblock_name].cpu().state_dict()
        part_model_params = self.model.cpu().state_dict()
        tmp_params = dict()
        for name in part_model_params:
            if "weight" in name:
                tmp = part_model_params[name].numpy()
                tmp_mask = np.where(np.abs(tmp) >= self.args.defdyexit_binarize_threshold, 1, 0)
                tmp_mask_params = torch.from_numpy(tmp * tmp_mask)
                # self.binarize_mask[exitblock_name+"."+name]=tmp_mask
                # tmp_params[exitblock_name+"."+name] = tmp_mask_params
                self.binarize_mask[name] = tmp_mask
                tmp_params[name] = tmp_mask_params
        return tmp_params

    def cal_model_part_params_nums(self):
        model=copy.deepcopy(self.model)
        total_params = 0
        for name, parameter in model.cpu().named_parameters():
            if "weight" in name:
                tmp = parameter.detach().numpy()
                tmp_mask = np.where(np.abs(tmp) >= self.args.defdyexit_binarize_threshold, 1, 0)
                total_params += np.sum(tmp_mask)
        return total_params / 1e6

    # 设置模型的部分参数
    def set_model_part_params(self, model_parameters):
        all_model_params = self.model.cpu().state_dict()
        for layer_name in model_parameters.keys():
            #
            old_layer = all_model_params[layer_name].numpy()
            old_binarize_mask_OP = np.where(self.binarize_mask[layer_name] > 0, 0, 1)
            old_layer_mask = old_layer * old_binarize_mask_OP

            new_layer = model_parameters[layer_name].numpy()
            new_layer_mask = new_layer * self.binarize_mask[layer_name]
            tmp_params = old_layer_mask + new_layer_mask

            all_model_params[layer_name] = torch.from_numpy(tmp_params)
        self.model.load_state_dict(all_model_params)


    def node_train(self, data=None):

        self.model.set_train_mode(2)
        self.model.set_exit_threshold(self.args.defdyexit_exit_threshold)
        self.model.set_exit_loss_weights(self.args.defdyexit_exit_loss_weights)
        self.model.to(self.device)
        self.model.train()

        if data is None:
            train_data_loader = self.train_data_loader
        else:
            train_data_loader = data

        criterion = self.criterion_fun

        optimizer = self.optimizer_fun(self.model.parameters(), lr=self.args.lr, momentum= self.args.momentum)

        for epoch in range(self.args.epochs):
            for batch_idx, (x, labels) in enumerate(train_data_loader):
                x = x.to(self.device)

                labels = labels.type(torch.LongTensor)
                labels = labels.to(self.device)


                self.model.zero_grad()

                log_probs = self.model(x)
                rawloss = [criterion(output, labels) for output in log_probs]

                losses = []
                tmp_losses = []

                for l, w in zip(rawloss, self.model.exit_loss_weights):
                    losses.append(w * l)
                    tmp_losses.append(w * l.item())

                for loss in losses[:-1]:
                    loss.backward(retain_graph=True)
                losses[-1].backward()

                optimizer.step()


    def node_test(self, data=None):

        self.model.set_train_mode(3)
        self.model.set_exit_threshold(self.args.defdyexit_exit_threshold)
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
                x = x.to(self.device)

                labels = labels.type(torch.LongTensor)
                labels = labels.to(self.device)
                self.model.zero_grad()
                log_probs = self.model(x)

                loss = criterion(log_probs, labels)
                if np.isnan(loss.item()):
                    result["loss"].append(0)
                else:
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

    parser.add_argument('--gpu_switch', type=bool, default=True)
    parser.add_argument('--gpu_id', type=int, default=0)

    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                                of optimizer")
    parser.add_argument('--criterion', type=str, default='cross', help="type \
                                    of optimizer")
    parser.add_argument('--epochs', type=int, default='5', help="type \
                                        of optimizer")
    parser.add_argument('--lr', type=float, default='0.001', help="type \
                                            of learning rate")
    parser.add_argument('--device_type', type=str, default="gpu",
                        help='gpu or cpu')

    parser.add_argument('--momentum', type=float, default='0.7', help="type \
                                                       of SGD momentum")

    # FedDyExit argument

    parser.add_argument('--defdyexit_binarize_threshold', type=float, default="0.001",
                        help='binarize threshold')

    parser.add_argument('--exitblock_name', type=str, default="exitblock", help="exit block name")

    parser.add_argument('--defdyexit_exit_threshold', type=float, default=0.05)

    dataname = "mnist"
    model=load_model("lenet",dataname)

    train_dataset, test_dataset = load_dataset(dataname)
    dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    a = Client_Node(args=parser.parse_args())

    a.set_model(model)

    a.node_test(dataloader)