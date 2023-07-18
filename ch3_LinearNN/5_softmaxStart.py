import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
import sys
sys.path.append("./")
from myDeepLearning import myPlot,myFun,myTools,myDataSetTools as myData

batch_size = 256
train_iter, test_iter = myData.load_data_fashion_mnist(batch_size)
#初始化神经网络特征
num_inputs = 28*28
num_outputs = 10
W = torch.normal(0,0.01,size=(num_inputs,num_outputs),requires_grad=True)
b = torch.zeros(num_outputs,requires_grad=True)

def net_softmax(X):
    return myFun.softmax(torch.matmul(X.reshape((-1,W.shape[0])),W)+b)

#交叉熵损失函数
def cross_entropy(y_hat,y):
    return - torch.log(y_hat[range(len(y_hat)),y])
lr = 0.1
def updater(batch_size):
    return torch.sgd([W,b],lr,batch_size)
num_epochs = 10
result = myTools.train_classify(net_softmax,train_iter,test_iter,cross_entropy,num_epochs,updater)
myPlot.show_linears(result,inputx=torch.Tensor([1,2,3,4,5,6,7,8,9,10]))
myTools.predict_classify(net_softmax,test_iter,n=8)
