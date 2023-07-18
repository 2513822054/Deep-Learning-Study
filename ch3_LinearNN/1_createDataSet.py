import random
import torch
import numpy as np
import os
from d2l import torch as d2l
import sys
sys.path.append("./")
from myDeepLearning import myPlot,myTools

def synthetic_data(w,b,num_examples):
    """生成y=Xw+b+噪声"""
    X = torch.normal(0,1,(num_examples,len(w)))
    Y = torch.matmul(X,w)+b
    Y += torch.normal(0,0.01,Y.shape)
    return X, Y.reshape((-1,1))
#生成数据集
true_w = torch.tensor([2,-3.4])
true_b = 4.2
inputs,outputs = synthetic_data(true_w,true_b,1000)
#myPlot.show_scatters(inputs[:,0:1].numpy(),outputs.numpy())                               #显示数据
myPlot.show_scatters3D(inputs[:,0:1].numpy(),inputs[:,1:2].numpy(),outputs.numpy())       #显示三维数据

#创建数据集     数据集一般用csv格式文件保存，csv格式以逗号隔开各数据 以\n隔开行
myTools.create_dataset(inputs,outputs,filename="3_2LinearRegression.csv")