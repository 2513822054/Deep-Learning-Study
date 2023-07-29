import torch,math,torchvision,sys
import pandas as pd
import numpy as np
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
from torch import nn
sys.path.append("./")
from myDeepLearning import myPlot,myUpdater,myTools,myDataSetTools as myData,myModel

#下载数据集
#print(myData.download_kaggle('kaggle competitions download -c house-prices-advanced-regression-techniques',deleteZip=True))
folderdata = {'foldername': 'house-prices-advanced-regression-techniques', 'filename': ['data_description.txt', 'sample_submission.csv', 'test.csv', 'train.csv']}
path = 'G:\\GraduateDoc\\acadamic\\pytorchLearning\\DataSet\\learning\\' + folderdata["foldername"] + '\\' 
data_pandas = myData.read_dataset_pandas(path,folderdata['filename'][3])

#按照每一列元素分类的个数排序将各分类映射到实数上
print(pd.get_dummies(data_pandas))
data_dummy = pd.get_dummies(data_pandas).values
print(data_dummy[0:1,-2:-1])
train_data = torch.Tensor(data_dummy)