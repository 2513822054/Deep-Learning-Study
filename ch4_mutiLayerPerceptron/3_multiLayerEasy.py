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

net = nn.Sequential(nn.Flatten(),nn.Linear(784,256),nn.ReLU(),nn.Linear(256,10))
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
net.apply(init_weights)                      #可以自定义一个 init_weights 函数，通过 net.apply(init_weights) 来初始化模型权重。

batch_size,lr,num_epochs = 256,1,100
loss = nn.CrossEntropyLoss(reduction='none')
trainer=torch.optim.SGD(net.parameters(),lr=lr)
train_iter , test_iter = myData.load_data_fashion_mnist(batch_size)
result = myTools.train_classify(net,train_iter,test_iter,loss,num_epochs,trainer)
myData.create_dataset(result[:,0:2],result[:,2:3],filename = "4_3_trainResult.csv",headname = "train_acc,test_acc,loss")
myPlot.show_linears(result,legend=['train_acc','test_acc','loss'],xlim=[1,100])
myTools.predict_classify(net,test_iter)