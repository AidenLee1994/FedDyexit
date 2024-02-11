# -*- coding:utf-8 -*-
# date: 2023/5/23 下午2:54
import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../")))
sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))



def load_model(model_name=None,dataset_name=None):

    if model_name == "eelenet":
        from model.new_lenet import Lenet
        model_class=Lenet(dataset_name)
    elif model_name == "lenet":
        from model.lenet import Lenet
        model_class = Lenet(dataset_name)
    elif model_name == "eeresnet":
        from model.new_resnet import Resnet
        model_class = Resnet(dataset_name)
    elif model_name == "resnet":
        from model.resnet import ResNet9
        model_class = ResNet9(dataset_name=dataset_name)

    else:
        model_class = None

    return model_class

if __name__ == '__main__':
    model=load_model("eeresnet","fmnist")
    print(model)