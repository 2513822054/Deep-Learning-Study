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
#按照每一列元素分类拆分成每个分类一列
inputs,outputs = data_pandas.iloc[:,1:-1],data_pandas.iloc[:,-1:]
data_dummy = pd.get_dummies(inputs,dummy_na=True)
for columnName in data_dummy:          #对列进行迭代
    data_dummy[columnName] = data_dummy[columnName].astype(float)
#获取训练数据的输入和输出
inputs = torch.Tensor(data_dummy.values[:,1:])
outputs = torch.Tensor(outputs.values)

inputs_train,inputs_test = inputs[:1000],inputs[1000:]
outputs_train,outputs_test = outputs[:1000],outputs[1000:]

lr = 0.1
batch_size,num_epochs = 100,100
net = nn.Sequential(nn.Linear(329,79),nn.ReLU(),nn.Linear(79,79),nn.Relu(),nn.Linear(79,1))  #定义模型
loss = nn.MSELoss()                                                                          #定义损失函数
trainer=torch.optim.SGD(net.parameters(),lr=lr)                                              #定义优化方法         

def init_weights(m): #初始化参数
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.1)
net.apply(init_weights)

inputs_train,outputs_train = myTools.data_batch(inputs,outputs,batch_size=batch_size,israndom=False)
def train(inputs_train,outputs_train,net,loss,trainer,inputs_test,outputs_test):
    for epoch in range(num_epochs):
        for train_iter in myTools.data_batch(inputs,outputs,batch_size=batch_size,israndom=False):
            myTools.train_epoch()