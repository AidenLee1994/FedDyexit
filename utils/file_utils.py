import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "../")))
sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../")))
from utils.path_utils import getroot_path
import os


class res_log_folder(object):
    def __init__(self,framework_name=None,log_save_path=None):
        self.framework_name=framework_name
        self.rootpath=getroot_path()+"/"
        if log_save_path is None:
            self.log_save_path = "res_log/"
        else:
            if log_save_path[-1] == "/":
                self.log_save_path = log_save_path
            else:
                self.log_save_path = log_save_path + "/"

        self.folder_path=self.rootpath + self.log_save_path + self.framework_name

        if self.check_folder_exist() is True:
            pass
        else:
            self.create_folder()



    def create_folder(self):
        os.makedirs(self.folder_path)

    def check_folder_exist(self):
        return os.path.exists(self.folder_path)

if __name__ == '__main__':
    res_log_folder("fedavg","test_path")