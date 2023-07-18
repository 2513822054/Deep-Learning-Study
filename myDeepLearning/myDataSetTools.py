import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
import sys
from myDeepLearning import myPlot,myTools
import random
import pandas as pd
import os

def data_batch(inputs,outputs,batch_size,israndom = True):
    '''将数据集分为多个batch'''
    allnum = len(inputs)
    '''Error checking'''
    if allnum != len(outputs):
        raise ValueError("inputs and outputs must be the same length!!!")
    '''Error checking end'''
    indices = list(range(allnum))
    if israndom:
        random.shuffle(indices)
    for i in range(0,allnum,batch_size):
        batch_indices = torch.tensor(indices[i:min(i+batch_size,allnum)])
        yield inputs[batch_indices],outputs[batch_indices]

def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator.

    Defined in :numref:`sec_linear_concise`"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def create_dataset(inputs,outputs,relativePath = "..",docname = "data",filename = "dataset.csv",headname = ""):
    #创建数据集     数据集一般用csv格式文件保存，csv格式以逗号隔开各数据 以\n隔开行
    shape1 = inputs.shape
    shape2 = outputs.shape
    if shape1[0] != shape2[0]:
        raise ValueError("shape1 and shape2 must be the same length.")
    os.makedirs(os.path.join(relativePath,docname),exist_ok=True)
    data_file = os.path.join(relativePath,docname,filename)
    with open(data_file,'w') as f:
        if headname == '':
            wtstr = ""
            for i in range(shape1[1]):
                wtstr += 'x'+str(i)+','
            for i in range(shape2[1]):
                wtstr += 'y'+str(i)+','
            wtstr = list(wtstr)
            wtstr[-1] = '\n'
            wtstr = ''.join(wtstr)
            f.write(wtstr)
        else:
            f.write(headname+'\n')
        for i in range(shape1[0]):
            wtstr = ''
            for j in range(shape1[1]):
                wtstr += '%f,'%inputs[i][j]
            for j in range(shape2[1]):
                wtstr += '%f,'%outputs[i][j]
            wtstr = list(wtstr)
            wtstr[-1] = '\n'
            wtstr = ''.join(wtstr)
            f.write(wtstr)

def read_dataset(inputnum,relativePath = "..",docname = "data",filename = "dataset.csv"):
    '''
    读取数据集
    inputnums表示输入个数,即数据集每一项输入的维度
    '''
    data_file = os.path.join(relativePath,docname,filename)
    data=pd.read_csv(data_file)
    return torch.Tensor(data.iloc[:,0:inputnum].values),torch.Tensor(data.iloc[:,inputnum:data.shape[1]].values)

def load_data_fashion_mnist(batch_size,resize = None):
    '''下载fashion数据集 加载到dataloader中并返回'''
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0,transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root='../data',train=True,transform = trans,download = True)
    mnist_test = torchvision.datasets.FashionMNIST(root='../data',train=False,transform = trans,download = True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,num_workers=0),
            data.DataLoader(mnist_test, batch_size, shuffle=False,num_workers=0))