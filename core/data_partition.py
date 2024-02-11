# -*- coding:utf-8 -*-
# date: 2023/5/23 下午3:17
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(__file__, "../")))
sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))


#Each client has same number of samples, and same distribution for all class samples.
#客户机数据数量和标签数量平均分配
def balanced_iid(dataset=None,num_client=None,seed=2023):
    #每个客户机样本数量一样多  对应的标签样本一样多
    np.random.seed(seed)

    labels = dataset.targets
    num_samples = len(labels)

    num_samples_per_client = int(num_samples / num_client)
    client_sample_nums = (np.ones(num_client) * num_samples_per_client).astype(
        int)

    rand_perm = np.random.permutation(num_samples)
    num_cumsum = np.cumsum(client_sample_nums).astype(int)
    client_indices_pairs = [(cid, idxs) for cid, idxs in
                            enumerate(np.split(rand_perm, num_cumsum)[:-1])]
    client_data_idxs_dict = dict(client_indices_pairs)
    return client_data_idxs_dict


#Assign different sample number for each client using Log-Normal distribution ,log-n(0,sgm^2)
# while keep same distribution for different class samples.
#客户机数据数量采用lognormal的形式分配，但每个客户机的标签分布平均
def unlablanced_iid(dataset=None,num_client=None,unbalance_sgm=0.0,seed=2023):
    np.random.seed(seed)
    #sgm接近于0时，会变成平衡分布
    if unbalance_sgm == 0.0:
        client_data_idxs_dict = balanced_iid(dataset=dataset,num_client=num_client,seed=seed)
    else:
        labels = dataset.targets
        num_samples=len(labels)
        balanced_num_samples_per_client = int(num_samples / num_client)
        client_sample_nums = np.random.lognormal(mean=np.log(balanced_num_samples_per_client),
                                                 sigma=unbalance_sgm,
                                                 size=num_client)
        client_sample_nums = (
                client_sample_nums / np.sum(client_sample_nums) * num_samples).astype(int)
        diff = np.sum(client_sample_nums) - num_samples  # diff <= 0

        # Add/Subtract the excess number starting from first client
        if diff != 0:
            for cid in range(num_client):
                if client_sample_nums[cid] > diff:
                    client_sample_nums[cid] -= diff
                    break

        rand_perm = np.random.permutation(num_samples)
        num_cumsum = np.cumsum(client_sample_nums).astype(int)
        client_indices_pairs = [(cid, idxs) for cid, idxs in
                                enumerate(np.split(rand_perm, num_cumsum)[:-1])]
        client_data_idxs_dict = dict(client_indices_pairs)

    return client_data_idxs_dict

#客户机数据数量采用平均的形式分配，但每个客户机的标签按照迪利克雷分配
def balance_dirichlet(dataset=None,num_client=None,dir_alpha=0.5,seed=2023):
    # 每个客户机样本数量一样多  对应的标签样本一样多
    np.random.seed(seed)

    labels = np.array(dataset.targets)
    num_samples = len(labels)
    num_classes = len(set(labels))


    num_samples_per_client = int(num_samples / num_client)
    client_sample_nums = (np.ones(num_client) * num_samples_per_client).astype(
        int)

    rand_perm = np.random.permutation(labels.shape[0])
    targets = labels[rand_perm]

    class_priors = np.random.dirichlet(alpha=[dir_alpha] * num_classes,
                                       size=num_client)
    prior_cumsum = np.cumsum(class_priors, axis=1)
    idx_list = [np.where(targets == i)[0] for i in range(num_classes)]
    class_amount = [len(idx_list[i]) for i in range(num_classes)]

    client_indices = [np.zeros(client_sample_nums[cid]).astype(np.int64) for cid in
                      range(num_client)]

    while np.sum(client_sample_nums) != 0:
        curr_cid = np.random.randint(num_client)
        # If current node is full resample a client
        if client_sample_nums[curr_cid] <= 0:
            continue
        client_sample_nums[curr_cid] -= 1
        curr_prior = prior_cumsum[curr_cid]
        while True:
            curr_class = np.argmax(np.random.uniform() <= curr_prior)
            # Redraw class label if no rest in current class samples
            if class_amount[curr_class] <= 0:
                continue
            class_amount[curr_class] -= 1
            client_indices[curr_cid][client_sample_nums[curr_cid]] = \
                idx_list[curr_class][class_amount[curr_class]]

            break

    client_data_idxs_dict = {cid: client_indices[cid] for cid in range(num_client)}
    return client_data_idxs_dict

#客户机数据数量采用lognormal的形式分配，但每个客户机的标签按照迪利克雷分配。
def unbalanced_dirichlet(dataset=None,num_client=None,dir_alpha=0.5,unbalance_sgm=0.0,seed=2023):
    np.random.seed(seed)
    # sgm接近于0时，会变成平衡分布
    if unbalance_sgm == 0.0:
        client_data_idxs_dict = balance_dirichlet(dataset=dataset,num_client=num_client,dir_alpha=dir_alpha,seed=seed)
    else:

        labels = np.array(dataset.targets)
        num_samples = len(labels)
        num_classes = len(set(labels))

        balanced_num_samples_per_client = int(num_samples / num_client)
        client_sample_nums = np.random.lognormal(mean=np.log(balanced_num_samples_per_client),
                                                 sigma=unbalance_sgm,
                                                 size=num_client)
        client_sample_nums = (
                client_sample_nums / np.sum(client_sample_nums) * num_samples).astype(int)
        diff = np.sum(client_sample_nums) - num_samples  # diff <= 0

        # Add/Subtract the excess number starting from first client
        if diff != 0:
            for cid in range(num_client):
                if client_sample_nums[cid] > diff:
                    client_sample_nums[cid] -= diff
                    break

        rand_perm = np.random.permutation(labels.shape[0])
        targets = labels[rand_perm]

        class_priors = np.random.dirichlet(alpha=[dir_alpha] * num_classes,
                                           size=num_client)
        prior_cumsum = np.cumsum(class_priors, axis=1)
        idx_list = [np.where(targets == i)[0] for i in range(num_classes)]
        class_amount = [len(idx_list[i]) for i in range(num_classes)]

        client_indices = [np.zeros(client_sample_nums[cid]).astype(np.int64) for cid in
                          range(num_client)]

        while np.sum(client_sample_nums) != 0:
            curr_cid = np.random.randint(num_client)
            # If current node is full resample a client
            if client_sample_nums[curr_cid] <= 0:
                continue
            client_sample_nums[curr_cid] -= 1
            curr_prior = prior_cumsum[curr_cid]
            while True:
                curr_class = np.argmax(np.random.uniform() <= curr_prior)
                # Redraw class label if no rest in current class samples
                if class_amount[curr_class] <= 0:
                    continue
                class_amount[curr_class] -= 1
                client_indices[curr_cid][client_sample_nums[curr_cid]] = \
                    idx_list[curr_class][class_amount[curr_class]]

                break

        client_data_idxs_dict = {cid: client_indices[cid] for cid in range(num_client)}
    return client_data_idxs_dict


def hetero_dirichlet_partition(dataset=None,num_client=None,dir_alpha=0.5,min_require_sample_size=None,seed=2023):
    """
    Non-iid partition based on Dirichlet distribution. The method is from "hetero-dir" partition of
    `Bayesian Nonparametric Federated Learning of Neural Networks <https://arxiv.org/abs/1905.12022>`_
    and `Federated Learning with Matched Averaging <https://arxiv.org/abs/2002.06440>`_.
    This method simulates heterogeneous partition for which number of data points and class
    proportions are unbalanced. Samples will be partitioned into :math:`J` clients by sampling
    :math:`p_k \sim \\text{Dir}_{J}({\\alpha})` and allocating a :math:`p_{p,j}` proportion of the
    samples of class :math:`k` to local client :math:`j`.
    Sample number for each client is decided in this function.
    Args:
        targets (list or numpy.ndarray): Sample targets. Unshuffled preferred.
        num_clients (int): Number of clients for partition.
        num_classes (int): Number of classes in samples.
        dir_alpha (float): Parameter alpha for Dirichlet distribution.
        min_require_size (int, optional): Minimum required sample number for each client. If set to ``None``, then equals to ``num_classes``.
    Returns:
        dict: ``{ client_id: indices}``.
    """
    np.random.seed(seed)
    labels = np.array(dataset.targets)
    num_classes = len(set(labels))

    if min_require_sample_size is None:
        min_require_sample_size = num_classes

    num_samples = labels.shape[0]

    min_size = 0
    while min_size < min_require_sample_size:
        idx_batch = [[] for _ in range(num_client)]
        # for each class in the dataset
        for k in range(num_classes):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(
                np.repeat(dir_alpha, num_client))
            # Balance
            proportions = np.array(
                [p * (len(idx_j) < num_samples / num_client) for p, idx_j in
                 zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in
                         zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    client_data_idxs_dict = dict()
    for cid in range(num_client):
        np.random.shuffle(idx_batch[cid])
        client_data_idxs_dict[cid] = np.array(idx_batch[cid])

    return client_data_idxs_dict



def paratition_data_index(paratition_type="home",dataset=None,num_client=1,unbalance_sgm=0,dir_alpha=0.5,min_require_sample_size=None,seed=2023):
    client_data_idxs_dict=None
    if paratition_type == "balance_iid":
        client_data_idxs_dict=balanced_iid(dataset,num_client,seed)
    elif paratition_type == "unlablanced_iid":
        client_data_idxs_dict=unlablanced_iid(dataset, num_client, unbalance_sgm, seed)
    elif paratition_type == "balance_dirichlet":
        client_data_idxs_dict=balance_dirichlet(dataset, num_client, dir_alpha, seed)
    elif paratition_type == "unbalance_dirichlet":
        client_data_idxs_dict=unbalanced_dirichlet(dataset,num_client,dir_alpha,unbalance_sgm,seed)
    elif paratition_type == "hetero_dirichlet":
        client_data_idxs_dict=hetero_dirichlet_partition(dataset,num_client,dir_alpha,min_require_sample_size,seed)
    
    return client_data_idxs_dict

if __name__ == '__main__':
    pass
    # from utils.data_utils import load_dataset
    # a,b=load_dataset("fmnist")
    # num=100
    # dic=balanced_iid(dataset=b, num_client=num, seed=2023)
    # dic=unlablanced_iid(dataset=b, num_client=num, unbalance_sgm=0.5, seed=2023)
    # dic = balance_dirichlet(dataset=b, num_client=num, dir_alpha=0.5, seed=2023)
    # dic = unbalanced_dirichlet(dataset=b, num_client=num, dir_alpha=0.5, unbalance_sgm=0.5, seed=2023)
    # dic = hetero_dirichlet_partition(dataset=b, num_client=num, dir_alpha=0.5, min_require_sample_size=20, seed=2023)
   
    # print(dic)

