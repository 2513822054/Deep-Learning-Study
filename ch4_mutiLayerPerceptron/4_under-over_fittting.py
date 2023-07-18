import torch,math,torchvision,sys
import numpy as np
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
from torch import nn
sys.path.append("./")
from myDeepLearning import myPlot,myFun,myTools,myDataSetTools as myData

# 生成多项式数据集
max_degree = 20  # 多项式的最大阶数
n_train, n_test = 100, 100  # 训练和测试数据集大小
true_w = np.zeros(max_degree)  # 分配大量的空间
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features)                                                  #洗牌
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))     #算出x的n次方
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)  # gamma(n)=(n-1)!
# labels的维度:(n_train+n_test,)
labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)                     #噪声项

# NumPy ndarray转换为tensor
true_w, features, poly_features, labels = [torch.tensor(x, dtype=torch.float32) for x in [true_w, features, poly_features, labels]]
# myPlot.show_scatters(features,labels)
# print(features.shape,poly_features.shape,labels.shape)
# fig = myPlot.myFigure(fabric=[1,2])
# fig.add_graph(labels,inputx=features,scatter=True,index=1)
# fig.add_graph(poly_features[:,:5],inputx=features,scatter=True,index=2)
# fig.show()

def train(train_features, test_features, train_labels, test_labels,num_epochs=1500):
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]
    # 不设置偏置，因为我们已经在多项式中实现了它
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = myData.load_array((train_features, train_labels.reshape(-1,1)),batch_size)
    test_iter = myData.load_array((test_features, test_labels.reshape(-1,1)),batch_size,is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    showData = []
    for epoch in range(num_epochs):
        myTools.train_epoc_classify(net, train_iter, loss, trainer)

        showData.append([myTools.evaluate_loss(net, train_iter, loss),myTools.evaluate_loss(net, test_iter, loss)])
    #myPlot.show_linears(torch.tensor(showData),inputx=torch.tensor(range(num_epochs))+1,xlim=[1,401],legend=['train_loss','test_loss'])
    print('weight:', net[0].weight.data.numpy())
    return torch.tensor(showData)

# 四个维度
showData1 = train(poly_features[:n_train, :4], poly_features[n_train:, :4],labels[:n_train], labels[n_train:])
# 从多项式特征中选择前2个维度，即1和x  欠拟合
showData2 = train(poly_features[:n_train,:2],poly_features[n_train:,:2],labels[:n_train],labels[n_train:])
# 从多项式特征中选取所有维度  过拟合
showData3 = train(poly_features[:n_train, :], poly_features[n_train:, :],labels[:n_train], labels[n_train:])
x = torch.tensor(range(1500))+1
fig = myPlot.myFigure(fabric=[1,3])
fig.add_graph(showData1,inputx = x,index=1,yscale='log')
fig.add_graph(showData2,inputx = x,index=2,yscale='log')
fig.add_graph(showData3,inputx = x,index=3,yscale='log')
fig.show()