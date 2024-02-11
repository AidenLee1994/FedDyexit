# -*- coding:utf-8 -*-
# date: 2023/5/29 11:32
# Author: 奈斯兔米特油
# WeChat: Mar20th94
# SCUT PhD.Stu
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def args_parser():

    parser = argparse.ArgumentParser()

    #gpu的使用
    # 数据集的名称
    parser.add_argument('--gpu_switch', type=str2bool, default=True)
    parser.add_argument('--gpu_id', type=int, default=0)

    # data info
    # 数据集的名称
    parser.add_argument('--dataset_name', type=str, default='mnist', help='dataset used for training')

    # 判断是使用数据的类型  iid  non-idd
    parser.add_argument("--paratition_type", type=str, default="balance_iid", help="paratition_type")

    # 计算时使用的数据尺寸
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size  (default: 32)')

    # 计算时使用的数据尺寸
    parser.add_argument('--noise_switch', type=str2bool, default=False,
                        help='wether add noise')

    # 计算时使用的数据尺寸
    parser.add_argument('--noise_level', type=float, default=0.5,
                        help='noise level')

    # 数据集分给客户的切片，在non iid时需要使用
    parser.add_argument('--unbalance_sgm', type=float, default=0.5,
                        help='unbalance sgm')

    parser.add_argument('--dir_alpha', type=float, default=0.5,
                        help='dir sgm')

    # 数据集分给客户的切片，需要给客户端分配的样本标签数量
    parser.add_argument('--data_different_label_number', type=float, default=1, help='data different_label_number')

    # 数据集分给客户的切片，需要给客户端分配的样本多少
    parser.add_argument('--min_require_sample_size', type=int, help='min require sample size')

    # 计算时使用的数据尺寸
    parser.add_argument('--seed', type=int, default=2022,
                        help='seed')

    # model info
    parser.add_argument('--model_name', type=str, default='eeresnet', help="model name")

    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                                of optimizer")
    parser.add_argument('--criterion', type=str, default='cross', help="type \
                                    of optimizer")

    parser.add_argument('--epochs', type=int, default='5', help="type \
                                        of optimizer")
    parser.add_argument('--lr', type=float, default='0.001', help="type \
                                            of learning rate")
    parser.add_argument('--momentum', type=float, default='0.95', help="type \
                                               of SGD momentum")



    # client info
    # 参与训练的客户机总数
    parser.add_argument('--client_num_in_total', type=int, default=100, metavar='N',
                        help='number of clients in a distributed cluster')

    # 单次通讯中允许的通讯客户机数量
    parser.add_argument('--client_num_per_round', type=int, default=5, metavar='N',
                        help='number of clients in training stage')

    # 一共要进行多少轮的客户机与服务器通讯
    parser.add_argument('--comm_round', type=int, default=100,
                        help='round number of communications ')

    #FedDyExit argument

    parser.add_argument('--defdyexit_binarize_threshold', type=float, default=0.01,
                        help='binarize threshold')

    parser.add_argument('--exitblock_name', type=str, default="exitblock", help="exit block name")

    parser.add_argument('--defdyexit_exit_threshold', type=float, default=0.05)
    parser.add_argument('--defdyexit_exit_loss_weights', type=float, default=0.5)




    #synFlowFL argument

    parser.add_argument('--synflowfl_pruning_ratio', type=float,default=0.3)

    parser.add_argument('--synflowfl_purning_epoch', type=int, default=50)

    parser.add_argument('--synflowfl_pre_train_epochs', type=int, default=1)



    # earlybird argument

    parser.add_argument('--early_bird_percent', type=float, default=0.4, help='early bird puring percent')

    parser.add_argument('--early_bird_epoch_keep', type=int, default=5, help='early bird epoch keep')

    parser.add_argument('--early_bird_local_epoch', type=int, default=5, help='early bird local epoch')

    parser.add_argument('--early_bird_finetune_lr', type=float, default=0.01, help='early_bird_finetune_lr')

    parser.add_argument('--early_bird_sparse_rate', type=float, default=0.0001,
                        help='scale sparse rate (default: 0.0001)')

    parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                        help='train with channel sparsity regularization')


    # per-fedavg argument

    parser.add_argument('--per_fedavg_alpha', type=float, default=0.001, help='per_fedavg_alpha')

    parser.add_argument('--per_fedavg_beta', type=float, default=0.001, help='per_fedavg_beta')

    # Ditto argument

    parser.add_argument('--ditto_mu', type=float, default=0, help="Proximal rate for Ditto")

    parser.add_argument('--ditto_pre_epochs',type=int, default=5, help = "Ditto pre local epochs")

    #FedPHP argument

    parser.add_argument('--fedphp_mu', type=float, default=0, help="Proximal rate for FedPHP")

    parser.add_argument('--fedphp_lamda', type=float, default=0.8, help="Regularization weight")

    #Fedbabu argument

    parser.add_argument('--fedbabu_post_epochs',type=int, default=5, help = "fedbaba post local epochs")

    #FedDistill argument
    parser.add_argument('--feddistill_lamda', type=float, default=1.0, help="Regularization weight")

    #pfedme argument

    parser.add_argument("--pfedme_lamda", type=float, default=1.0, help="Regularization weight")

    parser.add_argument( "--pfedme_K", type=int, default=5, help="Number of personalized training steps for pFedMe")

    parser.add_argument("--pfedme_p_learning_rate", type=float, default=0.01,
                        help="personalized learning rate to caculate theta aproximately using K steps")

    parser.add_argument("--pfedme_beta", type=float, default=0.0, help="Average moving parameter for pFedMe")

    parser.add_argument("--pfedme_learning_rate_decay_gamma", type=float, default=0.99)


    #fedprox argument

    parser.add_argument( "--fedprox_mu", type=float, default=0,help="Proximal rate for FedProx")

    parser.add_argument("--fedprox_learning_rate_decay_gamma", type=float, default=0.99)

    #apfl argument

    parser.add_argument( "--apfl_alpha", type=float, default=1.0)

    parser.add_argument("--apfl_learning_rate_decay_gamma", type=float, default=0.99)


    # fedmask argument

    parser.add_argument('--fedmask_binarize_threshold', type=float, default=0.01, help='fedmask_binarize_threshold')

    # lotteryfl argument
    parser.add_argument('--prune_percent', type=float, default=10,
                        help='pruning percent')
    parser.add_argument('--prune_start_acc', type=float, default=0.2,
                        help='pruning start acc')
    parser.add_argument('--prune_end_rate', type=float, default=0.5,
                        help='pruning end rate')
    parser.add_argument('--mask_ratio', type=float, default=0.5,
                        help='mask ratio')


    args = parser.parse_args()

    return args

if __name__ == '__main__':
    print(args_parser())