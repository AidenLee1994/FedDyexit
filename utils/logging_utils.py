# -*- coding:utf-8 -*-
# date: 2023/5/23 下午2:54
import logging
from datetime import datetime
import os

class Personal_Logger(object):
    def __init__(self,log_name=None,log_save_path="/res_log/"):
        self.log_name=log_name
        self.log_save_path=log_save_path
        if log_name is None:
            self.log_name=datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.logging=logging.getLogger(self.log_name)

        self.log_file=self.log_save_path+"/"+log_name+".log"
        self.check_log()

        self.logging.setLevel(logging.INFO)
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(message)s'))


        self.logging.addHandler(console)

        save_handler=logging.FileHandler(self.log_save_path+"/"+log_name+".log")
        save_handler.setLevel(logging.CRITICAL)
        save_handler.setFormatter(logging.Formatter("%(message)s"))

        self.logging.addHandler(save_handler)

    def check_log(self):
        if os.path.exists(self.log_file):
            os.remove(self.log_file)


def generate_log_file_name(framework_name,args):

    model_name=args.model_name
    model_optimizer=args.optimizer
    model_criterion=args.criterion

    train_epoch=args.epochs
    train_lr=args.lr
    train_client_num_in_total=args.client_num_in_total
    train_client_num_per_round=args.client_num_per_round
    train_comm_round=args.comm_round

    dataset_name = args.dataset_name
    dataset_paratition_type = args.paratition_type
    dataset_batch_size = args.batch_size
    dataset_noise_level = args.noise_level
    dataset_unbalance_sgm = args.unbalance_sgm
    dataset_dir_alpha = args.dir_alpha
    dataset_different_label_number = args.data_different_label_number
    dataset_min_require_sample_size = args.min_require_sample_size

    base_name="{}_{}_{}_{}_epoch{}_lr{}_clienttotal{}_clientper{}_round{}_{}_{}_batch{}_noise{}".format(framework_name,model_name,model_optimizer,model_criterion,
                                                            train_epoch,train_lr,train_client_num_in_total,
                                                            train_client_num_per_round,train_comm_round,dataset_name,
                              dataset_paratition_type,dataset_batch_size,dataset_noise_level)

    add_paratition_name=None
    if dataset_paratition_type == "balance_iid":
        add_paratition_name=""
    elif dataset_paratition_type == "unlablanced_iid":
        add_paratition_name = "{}".format(dataset_unbalance_sgm)
    elif dataset_paratition_type == "balance_dirichlet":
        add_paratition_name = "{}".format(dataset_dir_alpha)
    elif dataset_paratition_type == "unbalance_dirichlet":
        add_paratition_name = "{}_{}".format(dataset_dir_alpha,dataset_unbalance_sgm)
    elif dataset_paratition_type == "hetero_dirichlet":
        add_paratition_name = "{}_{}".format(dataset_dir_alpha,dataset_min_require_sample_size)
    elif dataset_paratition_type == "quantity_lable":
        add_paratition_name = "{}".format(dataset_different_label_number)

    if add_paratition_name == "":
        logname=base_name
    else:
        logname=base_name+"_"+add_paratition_name

    framework=framework_name.split("_")[0]
    if framework == "feddyexit":
        logname=logname+"_threshold"+str(args.defdyexit_exit_threshold)+"_weight"+str(args.defdyexit_exit_loss_weights)+"_binarize"+str(args.defdyexit_binarize_threshold)

    return logname


if __name__ == '__main__':
    # l=Personal_Logger("abc")
    #
    # l.logging.debug("This is a debug log.")
    # l.logging.critical("ThiWs is a critical log.")
    # l.logging.info("This is a info log.")
    # l.logging.warning("This is a warning log.")
    # l.logging.error("This is a error log.")
    # l.logging.critical("ThiRs is a critical log.")

    #test2
    from main.options import args_parser

    args = args_parser()
    a=generate_log_file_name("a",args)
    print(a)