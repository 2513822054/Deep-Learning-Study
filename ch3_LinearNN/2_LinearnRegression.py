import random
import torch
import numpy as np
import os
from d2l import torch as d2l
import sys
sys.path.append("./")
from myDeepLearning import myPlot,myTools

def Linreg(X,w,b):
    """定义线性回归模型  即单层感知机"""
    return torch.matmul(X,w)+b
def squared_loss(y_hat,y):
    """定义L2范式的损失函数"""
    return (y_hat - y.reshape(y_hat.shape))**2/2
def sgd(params,lr,batch_size):
    """随机梯度下降法优化模型"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
#获取数据集
inputs,outputs = myTools.read_dataset(2,filename="3_2LinearRegression.csv")

#初始化预测参数
w = torch.normal(0,0.01,size=(2,1),requires_grad=True)
b = torch.zeros(1,requires_grad=True)
#初始化训练参数
lr = 0.03       #训练率
num_epochs = 10  #训练轮数
net = Linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X,y in myTools.data_batch(inputs,outputs,10):
        X.requires_grad_(False)
        l = loss(net(X,w,b),y)       #计算模型和实际的损失
        l.sum().backward()           #损失反向传播
        sgd([w,b],lr,10)
    with torch.no_grad():
        train_l = loss(net(inputs,w,b),outputs)
        print(f"epoch{epoch+1},loss{float(train_l.mean()):f}")
