# -*- coding:utf-8 -*-
# date:2022/12/7 2:22

import os

def getroot_path():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return BASE_DIR


if __name__ == '__main__':
    print(getroot_path())