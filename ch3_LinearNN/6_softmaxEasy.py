import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
from torch import nn
import sys
sys.path.append("./")
from myDeepLearning import myPlot,myFun,myTools,myDataSetTools as myData

# batch_size = 256
# train_iter,test_iter = myData.load_data_fashion_mnist(batch_size)

# net = nn.Sequential(nn.Flatten(),nn.Linear(784,10))
# def init_weights(m):
#     if type(m) == nn.Linear:
#         nn.init.normal_(m.weight,std=0.01)
# net.apply(init_weights)

# loss = nn.CrossEntropyLoss(reduction='none')
# trainer = torch.optim.SGD(net.parameters(),lr=0.1)
# num_epochs=10
# result = myTools.train_classify(net,train_iter,test_iter,loss,num_epochs,trainer)
# #result = torch.Tensor([[11,22,333],[4,5,6],[7,8,9],[10,9,8]])
# myData.create_dataset(result[:,0:2],result[:,2:3],filename = "3_6_6result.csv",headname = "train_acc,test_acc,loss")
inputs,outputs = myData.read_dataset(2,filename = "3_6_6result.csv")
result = torch.cat((inputs,outputs),dim=1)
myPlot.show_linears(result,inputx = torch.Tensor([1,2,3,4,5,6,7,8,9,10]),xlim = [1,10],legend=['train_acc','test_acc','loss'])