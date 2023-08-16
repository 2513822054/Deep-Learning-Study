import torch,math,torchvision,sys
import pandas as pd
import numpy as np
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
from torch import nn
sys.path.append("./")
from myDeepLearning import myPlot,myUpdater,myTools,myDataSetTools as myData,myModel

lr = 0.001
batch_size,num_epochs = 10,100
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
data_dummy = data_dummy.fillna(0)
#获取训练数据的输入和输出
inputs = torch.Tensor(data_dummy.values[:,1:])
outputs = torch.Tensor(outputs.values)

inputs_train,inputs_test = inputs[:1300],inputs[1300:]
outputs_train,outputs_test = outputs[:1300],outputs[1300:]
train_data = data.TensorDataset(inputs_train,outputs_train)
test_data = data.TensorDataset(inputs_test,outputs_test)
train_iter = data.DataLoader(train_data,batch_size,shuffle=True)
test_iter = data.DataLoader(train_data,batch_size,shuffle=True)

#net = nn.Sequential(nn.Linear(329,79),nn.Sigmoid(),nn.Linear(79,79),nn.Sigmoid(),nn.Linear(79,1))  #定义模型
net = nn.Sequential(nn.Linear(329,110),nn.Tanh(),nn.Linear(110,110),nn.Tanh(),nn.Linear(110,1))
#loss = nn.MSELoss()                                                                          #定义损失函数
loss = nn.L1Loss(reduction='mean')
trainer=torch.optim.SGD(net.parameters(),lr=0.1)                                              #定义优化方法         

def init_weights(m): #初始化参数
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,mean=0,std=1)
        #nn.init.normal_(m.bias,mean=0,std=0)
        nn.init.constant_(m.bias,0)
net.apply(init_weights)

# net[0].weight.data.normal_(0,0.1)        #用高斯随机初始化权重参数
# net[0].bias.data.fill_(0)                 #用0偏置初始化偏置参数

def train(train_iter,test_iter,net,loss,trainer):
    for epoch in range(20):
        avgLoss = myTools.train_epoch(net,train_iter,loss,trainer)
        net.eval()
        y_hat = net(inputs_test)
        l = loss(y_hat, outputs_test)
        #print(l)
        l = l.mean().item()
        print("epoch:",epoch,"   loss:",avgLoss,"    test_loss:",l)
train(train_iter,test_iter,net,loss,trainer)
net.eval()
y_hat = net(inputs_test)
delta = (y_hat - outputs_test).abs()
print(inputs_test.shape)
print(y_hat.shape,outputs_test.shape,delta.shape)
print(torch.cat([outputs_test,y_hat,delta],dim=1))
#print(isinstance(trainer,torch.optim.Optimizer))