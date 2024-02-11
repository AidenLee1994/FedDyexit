# -*- coding:utf-8 -*-
# date:2023/6/1 14:44
# Author: 奈斯兔米特油
# WeChat: Mar20th94
# SCUT PhD.Stu

from utils.pytorch_utils import criterion_function,optimizer_function
from core.node_trainer import NodeTrainer
from collections import defaultdict



class Server_Node(NodeTrainer):

    def __init__(self, id=None,model=None,args=None,train_data_loader=None,test_data_loader=None):
        super(Server_Node, self).__init__(id=id, model=model, args=args, train_data_loader=train_data_loader, test_data_loader= test_data_loader)
        self.criterion_fun = criterion_function(args.criterion)
        self.optimizer_fun = optimizer_function(args.optimizer)
        self.num_classes=self.model.num_classes
        self.global_logits = [None for _ in range(self.num_classes)]

    # 聚合方法
    def aggregation(self, local_logits_list):
        agg_logits_label = defaultdict(list)
        for local_logits in local_logits_list:
            for label in local_logits.keys():
                agg_logits_label[label].append(local_logits[label])

        for [label, logit_list] in agg_logits_label.items():
            if len(logit_list) > 1:
                logit = 0 * logit_list[0].data
                for i in logit_list:
                    logit += i.data
                agg_logits_label[label] = logit / len(logit_list)
            else:
                agg_logits_label[label] = logit_list[0].data

        return agg_logits_label