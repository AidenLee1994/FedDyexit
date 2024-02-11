# -*- coding:utf-8 -*-
# date: 2023/5/23 下午2:51
from torch import nn
from torch import optim

def criterion_function(criterion_name="cross"):
    criterion_fun = None
    if criterion_name == "cross":
        criterion_fun = nn.CrossEntropyLoss()
    elif criterion_name == "mse":
        criterion_fun = nn.MSELoss()
    elif criterion_name == "margin":
        criterion_fun = nn.MultiMarginLoss()

    return criterion_fun

def optimizer_function(optimizer_name="sgd"):
    if optimizer_name == 'sgd':
        return optim.SGD
    elif optimizer_name == 'adagrad':
        return optim.Adagrad
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop
    elif optimizer_name == 'adadelta':
        return optim.Adadelta
    elif optimizer_name == 'adam':
        return optim.Adam
    elif optimizer_name == 'adamax':
        return optim.Adamax
    elif optimizer_name == 'sparseadam':
        return optim.SparseAdam
    elif optimizer_name == 'asgd':
        return optim.ASGD
    elif optimizer_name == 'rprop':
        return optim.Rprop
    elif optimizer_name == 'lbfgs':
        return optim.LBFGS