import torch
import random
import os
import pandas as pd
import matplotlib.pyplot as plt
from myDeepLearning import myPlot,myFun,myModel,myTimer
from torch.utils import data as torchData

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

#实用类 累加器
class Accumulator:
    """累加器讲解博客：https://www.cnblogs.com/zangwhe/p/17052548.html"""
    def __init__(self,n):
        self.data = [0.0]*n
    def add(self,*args):
        self.data = [a+float(b) for a,b in zip(self.data,args)]
    def reset(self):
        self.data = [0.0]*len(self.data)
    def __getitem__(self,idx):
        return self.data[idx]


def evaluate_loss(net, data_iter, loss):  #@save
    """评估给定数据集上的给定模型的给定损失算法的损失"""
    metric = Accumulator(2)  # 损失的总和,样本数量
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]

#计算精度
def accuracy(y_hat,y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] >1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())  

#评估精度
def evaluate_accuracy(net,data_iter):
    if isinstance(net,torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():               #下面只做模型评估 不优化模型  所以关闭反向求导
        for X,y in data_iter:
            metric.add(accuracy(net(X),y),y.numel())
        return metric[0]/metric[1]

def train_epoch(net,train_iter,loss,updater):
    '''训练一轮'''
    if isinstance(net,torch.nn.Module):
        net.train()
    for X,y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat,y)
        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])

def train_epoc_classify(net,train_iter,loss,updater):
    '''分类问题训练一轮'''
    #将模式设置为训练模式
    if isinstance(net,torch.nn.Module):
        net.train()
    #训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X,y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat,y)
        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()),accuracy(y_hat,y),y.numel())
    return metric[0]/metric[2],metric[1]/metric[2]              #第一个是损失，第二个是训练精确度

def train_classify(net,train_iter,test_iter,loss,num_epochs,updater):
    #animator = myPlot.Animator(xlabel='epoch',xlim=[1,num_epochs],ylim=[0.3,0.9],legend = ['train loss','tran acc','test acc'])
    metrics = []
    for epoch in range(num_epochs):
        train_metrics = train_epoc_classify(net,train_iter,loss,updater)
        test_acc = evaluate_accuracy(net,test_iter)
        #animator.add(epoch+1,train_metrics + (test_acc,))
        print('epoch:',epoch+1,'train_acc:',train_metrics[1],'test_acc:',test_acc,'loss:',train_metrics[0])
        metrics.append([train_metrics[0],test_acc,train_metrics[1]])
    metrics = torch.Tensor(metrics)
    train_loss,train_acc = train_metrics
    return metrics

def get_fashion_mnist_labels(labels):  #@save
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def predict_classify(net,test_iter,n=6):
    for X,y in test_iter:
        break
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' +pred for true,pred in zip(trues,preds)]
    myPlot.show_images(X[0:n].reshape(n,28,28),1,n,titles=titles[0:n])

def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声.
    Defined in :numref:`sec_linear_scratch`"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, torch.reshape(y, (-1, 1))

def load_array(data_arrays, batch_size, is_train=True):
    """创建一个pytorch迭代器
    Defined in :numref:`sec_linear_concise`"""
    dataset = torchData.TensorDataset(*data_arrays)
    return torchData.DataLoader(dataset, batch_size, shuffle=is_train)

def squared_loss(y_hat, y):
    """平方损失.
    Defined in :numref:`sec_linear_scratch`"""
    return (y_hat - torch.reshape(y, y_hat.shape)) ** 2 / 2