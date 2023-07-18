import torch
from torch.distributions import multinomial
from d2l import torch as d2l
import numpy as np
import sys
sys.path.append("./")
from myDeepLearning import myPlot

#扔骰子游戏
fair_prob = torch.ones(6)/6                              #fair prob 是一个概率分布列表，表示列表的下标被采样到的概率  此处设置列表每个值相等 表示扔骰子每个点的概率相等
#fair_prob = torch.Tensor([0.1,0.2,0.3,0.4])
counts = multinomial.Multinomial(10,fair_prob).sample([5000])    #每次采样十次时间，分500次采样
counts = counts.cumsum(dim = 0)                                 #计算累加值
count_sum = counts.sum(dim = 1,keepdims=True)                   #计算每次采样后的总事件数
prob = counts/count_sum                                         #计算每次采样时，事件的概率
print(prob)
myPlot.show_linears(prob.numpy(),title = "Title 1",xGrid=False)




#sample()是类Multinomial()中用来抽样的函数，仅接收一个参数 (sample_shape=torch.Size())，用来指定要抽样的次数，默认情况下仅抽样一次，输出一个形状为(len(probs), )的张量，否则，输出为(sample_shape, len(probs))的张量。