import torch
import numpy as np
import pandas as pd

#创建一个矩阵
A = torch.arange(9).reshape(3,3)
print(A)
print(A.T)                       #矩阵转置
B = A*A                          #矩阵暗元素乘法
print(B)
print(B+10)                      #矩阵+-*/标量，按元素加减乘除
print(B.sum())                   #求和
print(B.sum(axis=0))             #按第0轴（行轴）求和，行轴消失
print(B.sum(axis=1))             #按第1轴（列轴）求和，列轴消失
B=B.float()                      #将元素变为float类型
print(B.mean())                  #求平均
print(B.mean(axis=0))            #按第0轴求平均
print(B.mean(axis=0,keepdim=True))      #保持轴求平均
print("----------------------------------------------------------")
A = torch.ones(3,dtype=torch.float32)
B = torch.Tensor([3.0,7.0,9.0])
CC = torch.arange(9).reshape(3,3).float()
DD = torch.Tensor([[7.,8.,9.],[4.,5.,6.],[1.,2.,3.]])
print(torch.dot(A,B))                     #求向量点积
print(torch.mv(CC,B))                     #矩阵matrix向量vector积   列数必须和向量长度相同
print(torch.mm(CC,DD))                    #矩阵和矩阵乘法
print("----------------------------------------------------------")
print(A.norm())                           #A向量的第二范数(A的模长)
print(A.abs().sum())                      #A向量的第一范范数（绝对值求和）
print(CC.norm())                          #C矩阵的第二范数(所有元素平凡和开根号)