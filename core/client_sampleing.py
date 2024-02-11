# -*- coding:utf-8 -*-
# date: 2023/5/22 下午5:08
import numpy as np

# 选择client
def random_sampleing_clients(round_index,client_num_in_total, client_num_per_round):
    if client_num_per_round is None:
        print("Please set number of client for per round Train!")
    elif client_num_in_total < client_num_per_round:
        return np.array(range(1,client_num_in_total))
    else:
        np.random.seed(round_index)
        return np.random.choice(range(1,client_num_in_total), client_num_per_round, replace=False)