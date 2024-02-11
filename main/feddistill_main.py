# -*- coding:utf-8 -*-
# date:2023/6/1 15:58
# Author: 奈斯兔米特油
# WeChat: Mar20th94
# SCUT PhD.Stu
import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../")))
sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))

import copy
from options import args_parser
from utils.data_utils import get_paratition_data,DatasetSplit
from utils.model_utils import load_model

from torch.utils.data import DataLoader
from node.feddistill.server_node import Server_Node
from node.feddistill.client_node import Client_Node
from core.client_sampleing import random_sampleing_clients

from utils.logging_utils import Personal_Logger,generate_log_file_name
from time import time
from utils.file_utils import res_log_folder
import numpy as np

def main():


    args=args_parser()
    framework_name="feddistill"
    if args.noise_switch is True:
        noise_type = "noise"
    else:
        noise_type = "noiseless"
    framework_name = framework_name+"_"+noise_type

    folder=res_log_folder(framework_name)

    log_name = generate_log_file_name(framework_name, args)
    l = Personal_Logger(log_name,folder.folder_path)


    # load dataset and user groups
    train_dataset, test_dataset, user_groups = \
        get_paratition_data(paratition_type=args.paratition_type, dataset_name=args.dataset_name,
                            num_client=args.client_num_in_total, unbalance_sgm=args.unbalance_sgm,
                            dir_alpha=args.dir_alpha,
                            min_require_sample_size=args.min_require_sample_size,seed=args.seed)


    train_loader=DataLoader(DatasetSplit(dataset=train_dataset,idxs=user_groups[0],noise_switch=args.noise_switch,noise_level=args.noise_level,client_id=0,client_num=args.client_num_in_total),batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,shuffle=False)

    global_model=load_model(args.model_name,args.dataset_name)

    server=Server_Node(id=0, model=global_model, args=args,train_data_loader=train_loader,test_data_loader=test_loader)

    # init client
    l.logging.info("start init client!")
    client_list=dict()
    for i in range(1,args.client_num_in_total):

        train_loader=DataLoader(DatasetSplit(dataset=train_dataset,idxs=user_groups[i],noise_switch=args.noise_switch,noise_level=args.noise_level,client_id=i,client_num=args.client_num_in_total),
                                batch_size=args.batch_size)


        client_list[i]=Client_Node(id=i,args=args,
                                       train_data_loader=train_loader,
                                       test_data_loader=test_loader,
                                       model=copy.deepcopy(server.get_model()))


    train_time=0
    trans_time=0
    trans_sum = 0
    #start
    for round in range(args.comm_round):
        l.logging.info("start round num is : {}".format(round))
        client_params=[]

        # selected client
        selected_indx=random_sampleing_clients(round_index=round,client_num_in_total=args.client_num_in_total,client_num_per_round=args.client_num_per_round)
        l.logging.info("select client : {}".format(selected_indx))

        # # get server params
        # server_param = server.get_model_all_params()


        tmp_train_time=0
        tmp_trans_time=0
        for idx in selected_indx:
            client=client_list[idx]

            #update client params
            # client.set_model_all_params(copy.deepcopy(server_param))

            train_start_time=time()
            #train client
            client.node_train()
            train_end_time = time()
            tmp_train_time += train_end_time-train_start_time

            #get client params
            trans_start_time = time()
            client_param=client.get_logits()
            client_params.append(copy.deepcopy(client_param))
            trans_end_time = time()
            tmp_trans_time += (trans_end_time - trans_start_time)

        train_time+=tmp_train_time/len(selected_indx)
        trans_time+=tmp_trans_time/len(selected_indx)

        l.logging.info("training client success!")


        # cal server params
        server_params = server.aggregation(client_params)
        l.logging.info("server aggregation success!")
        # update server params

        l.logging.info("server model update success!")

        # update client params
        trans_update_start_time = time()
        for idx in selected_indx:
            client = client_list[idx]
            client.set_logits(server_params)
        trans_update_end_time = time()
        trans_time += (trans_update_end_time - trans_update_start_time) / len(selected_indx)

        l.logging.info("update client success!")

        loss = []
        pre = []
        acc = []
        recall = []
        f1 = []
        d_idx = []
        send_params_num = []

        for client_idx in selected_indx:
            res = client_list[client_idx].node_test(client_list[client_idx].train_data_loader)
            loss.append(res["loss"])
            pre.append(res["pre"])
            acc.append(res["acc"])
            recall.append(res["recall"])
            f1.append(res["f1"])
            d_idx.append(res["d_idx"])
            send_params_num.append(res["send_params_num"])
        trans_sum += np.mean(send_params_num)
        trans_sum += server.cal_model_all_params_nums(server.get_model())
        l.logging.critical(
            "acc:{:.4},pre:{:.4},f1:{:.4},recall:{:.4},d_idx:{:.4},loss:{:.4},send_params_num:{:.4},train_time:{:.4},trans_time:{:.4}".format(
                np.mean(acc), np.mean(pre), np.mean(f1), np.mean(recall),
                np.mean(d_idx), np.mean(loss), trans_sum, train_time, trans_time)
        )
    exit()


if __name__ == '__main__':
    main()