# tensor张量的基本操作（多维数组）
import torch
import numpy
#一个轴的张量是向量   两个轴的张量是矩阵
x = torch.arange(12)            #一个长度为12的向量（数组）
print(x.shape)                  #通过shape方法访问张量的形状（各个轴的长度）
print(x.numel())                #获取张量的元素总数

X = x.reshape(3,4)              #改变张量的形状，使向量变成矩阵
X = x.reshape(-1,4)             #参数为-1则会被自动计算出来

x = torch.zeros((2,3,4))        #创建一个全为0的轴长为2，3，4的张量
x = torch.ones((2,3,4))         #全为1
x = torch.randn(3,4)            #创建一个元素取值为均值0，方差1的高斯分布的3*4的矩阵
x = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])  #用python的list初始化tensor  外层对应轴0  内层对应轴1

#+ - * / ** 五个运算符可以直接用于tensor  其中**表示幂运算   tensor的结构必须相同
x = torch.tensor([1.0,2.0,4.0,8.0])
y = torch.tensor([2.0,2,2,2])
z = x**y

#cat方法   将两个张量按指定的轴合并
x = torch.arange(12,dtype = torch.float32).reshape((3,4))
y = torch.tensor([[12.0,11,10,9],[8,7,6,5],[4,3,2,1]])
z = torch.cat((x,y),dim=0)
z = x >= y               #两个张量逻辑运算 将bool结果放在对应位置上  z是一个3*4的bool类型的张量
z = x.sum                #将张量所有元素求和，产生一个单元素张量

#广播机制，用于处理不同形状的张量运算，会沿着长度为1的轴复制，直至两个矩阵形状相同
x = torch.tensor([[0],[2],[4],[6]])               #4*1
y = torch.tensor([[9,8]])                         #1*2
z = x + y                                         #x的1轴复制，y的0轴复制

X = torch.arange(25).reshape((5,5))
z1 = X[-1]                           #访问最后一行                          与X共享存储空间 X变了z1，z2，z3也会变
z2 = X[1:3]                          #访问第二行和第三行(下表从1-2，不包括3)  与X共享存储空间 X变了z1，z2，z3也会变
z3 = X[1:3,1:3]                      #访问第2，3行的第2，3列                 与X共享存储空间 X变了z1，z2，z3也会变
X[1:3,1:3] = 985                     #给X的第2，3行的第2，3列赋值
# print(z1)
# print(z2)
# print(z3)

x = torch.arange(12).reshape((3,4))
y = torch.arange(12).reshape((3,4))
y = x + y             #该操作会给y重新分配内存   但我们希望在y原有内存上原地操作
y[:] = x + y          #使用切片操作就可以将操作后的数据赋给y原来的内存 使得不需要开辟新的空间
y += x                #便捷赋值运算符也可达到同样的效果

