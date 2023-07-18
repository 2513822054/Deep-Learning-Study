import torch
import random
import os
import pandas as pd
import matplotlib.pyplot as plt
from myDeepLearning import myPlot,myFun,myModel,myTimer

def sgd(params,lr,batch_size):
    """随机梯度下降法优化模型"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()