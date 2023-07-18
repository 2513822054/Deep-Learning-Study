import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
import matplotlib.pyplot as plt
import sys
sys.path.append("./")
from myDeepLearning import myPlot,myTools

d2l.use_svg_display()
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root='../data',train=True,transform = trans,download = True)
mnist_test = torchvision.datasets.FashionMNIST(root='../data',train=False,transform = trans,download = True)

def get_fashion_mnist_labels(labels):  #@save
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

X, y = next(iter(data.DataLoader(mnist_train, batch_size=54)))
myPlot.show_images(X.reshape(54, 28, 28), 6, 9, titles=get_fashion_mnist_labels(y));