import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
import sys
from myDeepLearning import myPlot,myTools,myFun
import random
import pandas as pd
import os


def linreg(X, w, b):
    """线性回归模型
    Defined in :numref:`sec_linear_scratch`"""
    return torch.matmul(X, w) + b