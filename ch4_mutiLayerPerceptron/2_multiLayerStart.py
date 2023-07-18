import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
import matplotlib.pyplot as plt
from torch import nn
import sys
sys.path.append("./")
from myDeepLearning import myPlot,myFun,myTools,myDataSetTools as myData

batch_size = 256
train_iter,test_iter = myData.load_data_fashion_mnist(batch_size)

#初始化模型参数
num_inputs,num_outputs,num_hiddens = 784,10,256
W1 = nn.Parameter(torch.randn(num_inputs,num_hiddens,requires_grad=True)*0.01)  #第一层权重
b1 = nn.Parameter(torch.zeros(num_hiddens,requires_grad=True))                  #第一层偏置
W2 = nn.Parameter(torch.randn(num_hiddens,num_outputs,requires_grad=True)*0.01) #第二层权重
b2 = nn.Parameter(torch.zeros(num_outputs,requires_grad=True))

params = [W1,b1,W2,b2]
print(params)

def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X,a)

def net_relu(X):
    X = X.reshape(-1,num_inputs)
    H = relu(torch.mm(X,W1)+b1)
    return torch.mm(H,W2)+b2

loss = nn.CrossEntropyLoss(reduction='none')
num_epochs,lr=10,0.1
updater = torch.optim.SGD(params,lr=lr)
result = myTools.train_classify(net_relu,train_iter,test_iter,loss,num_epochs,updater)
myData.create_dataset(result[:,0:2],result[:,2:3],filename = "4_2_trainResult.csv",headname = "train_acc,test_acc,loss")
myPlot.show_linears(result)
myTools.predict_classify(net_relu,test_iter,legend=['train_acc','test_acc','loss'],xlim=[1,10])