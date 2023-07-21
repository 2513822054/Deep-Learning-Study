import torch,math,torchvision,sys
import numpy as np
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
from torch import nn
sys.path.append("./")
from myDeepLearning import myPlot,myUpdater,myTools,myDataSetTools as myData,myModel

myData.download_kaggle('kaggle competitions download -c house-prices-advanced-regression-techniques')