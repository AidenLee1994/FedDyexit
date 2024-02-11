#-*- coding:utf-8 -*-
import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "../")))
sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))

from torch.utils.data import Dataset
from utils.path_utils import getroot_path
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from sklearn.model_selection import train_test_split
class PACS(Dataset):
    def __init__(self,data_name=None,data_base_path=None,transform=None,target_transform=None):
        if data_name is None:
            self.data_name = "pacs"
        else:
            self.data_name=data_name
        if data_base_path is None:
            data_base_path=getroot_path()+"data/"
        self.file_path = data_base_path + self.data_name+"/"
        self.transform=transform
        self.target_transform=target_transform
        self.domains=["art_painting","cartoon","photo","sketch"]
        self.class2label={'dog':0,'elephant':1,'giraffe':2,'guitar':3,'horse':4,
                          'house':5,'person':6}
        self.processed_path=self.file_path+"processed/"
        self.data_path=self.processed_path+"data.pt"
        self.targets_path=self.processed_path+"targets.pt"
        self.data=[]
        self.targets=[]

        self.load_data()

    def load_data(self):
        if not os.path.exists(self.processed_path):
            os.mkdir(self.processed_path)
        if os.path.exists(self.data_path) and os.path.exists(self.targets_path):
            self.data=torch.load(self.data_path)
            self.targets=torch.load(self.targets_path)
        else:
            for domain in self.domains:
                for categrate in self.class2label.keys():
                    taget=self.class2label[categrate]
                    categrate_path=self.file_path+domain+"/"+categrate+"/"
                    image_name_list=os.listdir(categrate_path)
                    for image_name in image_name_list:
                        img=Image.open(categrate_path+image_name)
                        img=np.array(img)
                        if self.transform is not None:
                            img = self.transform(img)

                        self.data.append(img)
                        self.targets.append(torch.tensor(taget,dtype=torch.long))
            self.data=torch.stack(self.data)

            torch.save(self.data,self.data_path)
            torch.save(self.targets,self.targets_path)

    def __getitem__(self, index):
        img,target = self.data[index],self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img,target

    def __len__(self):
        return len(self.targets)

class data2pytorchdataset(Dataset):
    def __init__(self,data,targets):
        self.data=data
        self.targets=targets

    def __getitem__(self, item):
        return self.data[item],self.targets[item]

    def __len__(self):
        return len(self.targets)

def load_pacs(dataset_name="pacs",data_base_path=None):
    if data_base_path is None:
        data_base_path = getroot_path() + "data/"

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((32, 32)),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    pacs_data = PACS(data_name=dataset_name,data_base_path=data_base_path,transform=transform)
    x_train, x_test, y_train, y_test = train_test_split(pacs_data.data, pacs_data.targets, test_size=0.3,
                                                        random_state=2023)
    train_dataset = data2pytorchdataset(x_train, y_train)
    test_dataset = data2pytorchdataset(x_test, y_test)

    return train_dataset, test_dataset

