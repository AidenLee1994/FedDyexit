# -*- coding:utf-8 -*-
# date: 2023/5/23 下午2:55
from pycm import ConfusionMatrix
import numpy as np
import warnings
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore")

def dindex_score(origin_label,predict_label):
    cm=ConfusionMatrix(actual_vector=origin_label,predict_vector=predict_label)
    classes=cm.classes
    class_stats = cm.class_stat
    class_acc=class_stats["ACC"]
    class_sen = class_stats["TPR"]
    class_spe = class_stats["TNR"]
    d_indexs=[]
    for item in classes:
        acc=class_acc[item]
        if acc == "None":
            acc=0
        sen = class_sen[item]
        if sen == "None":
            sen = 0
        spe = class_spe[item]
        if spe == "None":
            spe = 0
        d_index=np.log2(1+acc)+np.log2(1+(sen+spe)/2)
        d_indexs.append(d_index)
    return np.mean(d_indexs)

def Evaluate(true_label, predict_label):

    acc = accuracy_score(true_label, predict_label)
    pre = precision_score(true_label, predict_label, average="macro")
    f1 = f1_score(true_label, predict_label, average="macro")
    recall = recall_score(true_label, predict_label, average="macro")
    d_idx= dindex_score(true_label, predict_label)
    res={'acc':acc,'pre':pre,'f1':f1,'recall':recall,'d_idx':d_idx}
    return res

if __name__ == '__main__':
    a = [1,3,4]
    b = [1, 0, 0]
    print(Evaluate(b, a))