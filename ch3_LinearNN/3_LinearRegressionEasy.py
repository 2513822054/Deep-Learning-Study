import random
import torch
from torch.utils import data
from torch import nn
import numpy as np
import os
from d2l import torch as d2l
import sys
sys.path.append("./")
from myDeepLearning import myPlot,myTools

#获取数据集
inputs,outputs = myTools.read_dataset(2,filename="3_2LinearRegression.csv")

#用dataloader 迭代batch获取数据集
def load_array(data_arrays,batch_size,is_train=True):
    '''构造一个pyTorch数据迭代器'''
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)
batch_size = 10
#构造一个数据集的迭代器，可以每次返回batch长度的数据
data_iter = load_array((inputs,outputs),batch_size)
print(data_iter.__len__)

#构造一个全连接神经网络
net = nn.Sequential(nn.Linear(2,1))       #规定输入输出形状
net[0].weight.data.normal_(0,0.01)        #用高斯随机初始化权重参数
net[0].bias.data.fill_(0)                 #用0偏置初始化偏置参数
loss = nn.MSELoss()                    #平方L2范式
trainer = torch.optim.SGD(net.parameters(),lr=0.03)  #定义SGD优化算法

num_epochs = 5
for epoch in range(num_epochs):
    for X,y in data_iter:
        l = loss(net(X),y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(inputs),outputs)
    print(f'epoch {epoch+1},loss {l:f}')
w = net[0].weight.data
b = net[0].bias.data
print(w,b)