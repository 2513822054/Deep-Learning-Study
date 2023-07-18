import torch
import numpy as np
import pandas as pd

X1 = torch.arange(5.)
X2 = torch.tensor([5.,4.,3.,2.,1.]) 
X1 = X1+1                            #创建两个向量
X1.requires_grad_(True)
X2.requires_grad_(True)              #设置两个张量的requires_grad属性为真，表示需要为该张量计算梯度
print(X1,X2)
y = 2 * torch.dot(X1,X1)
y.backward()
print(y,X1.grad)                     #计算X1*X1对X1的梯度

X1.grad.zero_()                      #将X1的梯度清零，防止梯度累加
y = torch.dot(X1,X2)
y.backward()
print(y,X1.grad,X2.grad)             #计算X1*X2对X1和X2的梯度

X1.grad.zero_()
X2.grad.zero_()
y=X1.sum()
y.backward()
print(y,X1.grad)                     #计算sum（X1）的梯度
