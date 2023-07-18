import torch,math,torchvision,sys
import numpy as np
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
from torch import nn
sys.path.append("./")
from myDeepLearning import myPlot,myUpdater,myTools,myDataSetTools as myData,myModel

num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
dropout1, dropout2 = 0.2, 0.5
dropout1 = 0.2
dropout2 = 0.3
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = myData.load_data_fashion_mnist(batch_size)

net1 = nn.Sequential(nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        # 在第一个全连接层之后添加一个dropout层
        nn.Dropout(dropout1),
        nn.Linear(256, 256),
        nn.ReLU(),
        # 在第二个全连接层之后添加一个dropout层
        nn.Dropout(dropout2),
        nn.Linear(256, 10))

net2 = nn.Sequential(nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net1.apply(init_weights);
net2.apply(init_weights);
trainer1 = torch.optim.SGD(net1.parameters(), lr=lr)
result1 = myTools.train_classify(net1, train_iter, test_iter, loss, num_epochs, trainer1)
trainer2 = torch.optim.SGD(net2.parameters(), lr=lr)
result2 = myTools.train_classify(net2, train_iter, test_iter, loss, num_epochs, trainer2)
fig = myPlot.myFigure(fabric=[1,2],suptitle='the Difference between nn with and without dropout regularization')
fig.add_graph(result1,legend=['train loss','train acc','test acc'],xlabel='epoch',index=1)
fig.add_graph(result2,legend=['train loss','train acc','test acc'],xlabel='epoch',index=2)
fig.show()