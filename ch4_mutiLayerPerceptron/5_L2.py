import torch,math,torchvision,sys
import numpy as np
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
from torch import nn
sys.path.append("./")
from myDeepLearning import myPlot,myUpdater,myTools,myDataSetTools as myData,myModel

# if torch.cuda.is_available():
#   torch.set_default_tensor_type(torch.cuda.FloatTensor)
#   print("using cuda:", torch.cuda.get_device_name(0))
#   pass

# device = torch.device("cuda" if torch.cuda.is_available else "cpu")

# print(device)


#生成数据
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = myTools.synthetic_data(true_w, true_b, n_train)
train_iter = myTools.load_array(train_data, batch_size)
test_data = myTools.synthetic_data(true_w, true_b, n_test)
test_iter = myTools.load_array(test_data, batch_size, is_train=False)

#从零开始实现
def init_params():
    '''初始化参数'''
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

def l2_penalty(w):
    '''定义L2范数惩罚'''
    return torch.sum(w.pow(2)) / 2

def train(lambd):
    w, b = init_params()
    net, loss = lambda X: myModel.linreg(X, w, b), myTools.squared_loss
    num_epochs, lr = 500, 0.003
    y_loss = [[],[]]
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 增加了L2范数惩罚项，
            # 广播机制使l2_penalty(w)成为一个长度为batch_size的向量
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            myUpdater.sgd([w, b], lr, batch_size)
        y_loss[0].append(myTools.evaluate_loss(net,train_iter,loss))
        y_loss[1].append(myTools.evaluate_loss(net,test_iter,loss))
    print('w的L2范数是：', torch.norm(w).item())
    return y_loss

def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')              #reduction=none    计算结果为向量形式为向量形式
    num_epochs, lr = 500, 0.003
    # 偏置参数没有衰减
    trainer = torch.optim.SGD([
        {"params":net[0].weight,'weight_decay': wd},
        {"params":net[0].bias}], lr=lr)
    y_loss = [[],[]]
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()
        y_loss[0].append(myTools.evaluate_loss(net,train_iter,loss))
        y_loss[1].append(myTools.evaluate_loss(net,test_iter,loss))
    print('w的L2范数：', net[0].weight.norm().item())
    return y_loss

y1 = train(lambd=0)
y2 = train(lambd=3)
y3 = train_concise(wd = 0)
y4 = train_concise(wd = 3)
fig = myPlot.myFigure(fabric=[2,2],suptitle="the Difference Loss Between NoneRegularization and L2Regularization")
fig.add_graph(torch.Tensor(y1),xlim=[1,500],dim=1,yscale='log',title='None Regularization',xlabel='epochs',ylabel='loss',legend=['train_loss','test_loss'])
fig.add_graph(torch.Tensor(y2),xlim=[1,500],index=2,dim=1,yscale='log',title='L2 Regularization',xlabel='epochs',ylabel='loss',legend=['train_loss','test_loss'])
fig.add_graph(torch.Tensor(y3),xlim=[1,500],index=3,dim=1,yscale='log',title='None Regularization',xlabel='epochs',ylabel='loss',legend=['train_loss','test_loss'])
fig.add_graph(torch.Tensor(y4),xlim=[1,500],index=4,dim=1,yscale='log',title='L2 Regularization',xlabel='epochs',ylabel='loss',legend=['train_loss','test_loss'])
fig.show()