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

figure = myPlot.myFigure(fabric=[3,2],figsize=[3*5,2*5],dpi = 100,suptitle="Main Perceptron")

#Relu
xRelu = torch.arange(-8.0,8.0,0.1,requires_grad=True)
yRelu = torch.relu(xRelu)
yRelu.backward(torch.ones_like(xRelu),retain_graph=True)
figure.add_graph(yRelu.detach(),inputx=xRelu.detach(),index=[1,1],title='Relu')
figure.add_graph(xRelu.grad,inputx=xRelu.detach(),index=[1,2],title='Relu Grad')

#sigmoid
xSig = torch.arange(-8.0,8.0,0.1,requires_grad=True)
ySig = torch.sigmoid(xSig)
ySig.backward(torch.ones_like(xSig),retain_graph=True)
figure.add_graph(ySig.detach(),inputx=xSig.detach(),index=[2,1],title='Sigmoid')
figure.add_graph(xSig.grad,inputx=xSig.detach(),index=[2,2],title='Sigmoid Grad')

#tanh
xTan = torch.arange(-8.0,8.0,0.1,requires_grad=True)
yTan = torch.tanh(xTan)
yTan.backward(torch.ones_like(xTan),retain_graph=True)
figure.add_graph(yTan.detach(),inputx=xTan.detach(),index=[3,1],title='Tanh')
figure.add_graph(xTan.grad,inputx=xTan.detach(),index=[3,2],title='Tanh Grad')

figure.show()