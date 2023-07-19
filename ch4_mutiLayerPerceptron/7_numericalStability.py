import torch,math,torchvision,sys
import numpy as np
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
from torch import nn
sys.path.append("./")
from myDeepLearning import myPlot,myUpdater,myTools,myDataSetTools as myData,myModel

x = torch.arange(-15.0, 15.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))
print(torch.stack([y.detach(),x.grad],dim=0).shape)
myPlot.show_linears(torch.stack([y.detach(),x.grad],dim=0),inputx=x.detach(),legend=['sigmoid','gradient'],dim=1,title='the sigmoid and its gradient')

M = torch.normal(0, 1, size=(4,4))
print('一个矩阵 \n',M)
for i in range(100):
    M = torch.mm(M,torch.normal(0, 1, size=(4, 4)))

print('乘以100个矩阵后\n', M)