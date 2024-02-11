import sys
import os

sys.path.append(os.path.abspath(os.path.join(__file__, "../")))
sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))

from utils.path_utils import getroot_path
from torch.utils.data import Dataset
import torch
from PIL import Image
# import jpeg4py as jpeg
# import cv2
from torchvision import transforms
import pandas as pd
import numpy as np
from tqdm import tqdm
from zipfile import ZipFile
from sklearn.model_selection import train_test_split


class DomaninNet(Dataset):
    def __init__(self, data_base_path=None, train=True):
        self.data_name = "domainnet"
        self.flag = train

        self.data_path = []
        self.targets = []
        self.folders = ['Painting', 'Quickdraw', 'Sketch', 'Infograph']

        if data_base_path is None:
            data_base_path = getroot_path() + "data/"
        self.file_path = data_base_path + self.data_name + "/"

        self.processed_path = self.file_path + "processed/"
        self.deal_processed_folder()

        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize((32,32)),
                                             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        self.data_path_targets = None

        if self.flag is True:
            self.data_path_targets_path = self.processed_path + "train_data_path_targets.csv"
        else:
            self.data_path_targets_path = self.processed_path + "test_data_path_targets.csv"

        self.deal_unzip()

        self.load_data_path_targets()

    def __getitem__(self, index):
        img_path = self.data_path[index]
        abs_img_path = self.file_path + img_path
        img = np.array(Image.open(abs_img_path).convert('RGB'))
        img = self.transform(img)
        target = torch.tensor(self.targets[index], dtype=torch.long)
        return img, target

    def __len__(self):
        return len(self.targets)

    def load_data_path_targets(self):
        if os.path.exists(self.data_path_targets_path):
            df = pd.read_csv(self.data_path_targets_path)
            self.data_path = df["data_path"]
            self.targets = df["targets"]

        else:
            if self.flag is True:
                self.scan_train_data()
            else:
                self.scan_test_data()

    def deal_processed_folder(self):
        path = self.processed_path
        if not os.path.exists(path):
            os.mkdir(path)

    def read_data_info(self, folder, path):
        df = pd.read_table(path, header=None, sep=" ")
        data = []
        target = []
        for _, row in df.iterrows():
            data.append(folder + "/" + row[0])
            target.append(row[1])
        return data, target

    def unzip_folder(self, zip_file_path, unzip_file_path, unzip_folder):
        if os.path.exists(unzip_file_path + unzip_folder):
            pass
        else:
            unzip_file = ZipFile(zip_file_path)
            unzip_file.extractall(unzip_file_path)

    def deal_unzip(self):
        for folder in self.folders:
            zip_file_path = self.file_path + folder + "/" + folder.lower() + ".zip"
            unzip_file_path = self.file_path + folder + "/"
            self.unzip_folder(zip_file_path, unzip_file_path, folder.lower())

    def scan_train_data(self):

        data_type = "train"

        for folder in tqdm(self.folders):
            list_path = self.file_path + folder + "/" + folder.lower() + "_" + data_type + ".txt"

            data_path, data_targets = self.read_data_info(folder, list_path)
            self.data_path.extend(data_path)
            self.targets.extend(data_targets)

        df = pd.DataFrame()
        df["data_path"] = self.data_path
        df["targets"] = self.targets
        df.to_csv(self.data_path_targets_path, index=False)

    def scan_test_data(self):
        data_type = 'test'
        for folder in tqdm(self.folders):
            list_path = self.file_path + folder + "/" + folder.lower() + "_" + data_type + ".txt"

            data_path, data_targets = self.read_data_info(folder, list_path)
            data_path = np.array(data_path)
            data_targets = np.array(data_targets)

            _, x_test, _, y_test = train_test_split(data_path, data_targets, test_size=0.1, random_state=2023)

            self.data_path.extend(x_test)
            self.targets.extend(y_test)

        df = pd.DataFrame()
        df["data_path"] = self.data_path
        df["targets"] = self.targets
        df.to_csv(self.data_path_targets_path, index=False)


def load_domainnet():
    train_dataset = DomaninNet(train=True)
    test_dataset = DomaninNet(train=False)

    return train_dataset, test_dataset


if __name__ == "__main__":
    a = DomaninNet(train=False)
    # print(a)


