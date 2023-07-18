import os
import pandas as pd
import torch


#创建数据集     数据集一般用csv格式文件保存，csv格式以逗号隔开各数据 以\n隔开行
os.makedirs(os.path.join('..','data'),exist_ok=True)
data_file = os.path.join('..','data','2_house_tiny.csv')
with open(data_file,'w') as f:
    f.write('NumRooms,Alley,Price\n')
    f.write('NA,PAVE,127500\n')
    f.write('3,NA,106000\n')
    f.write('1,NA,178100\n')
    f.write('NA,NA,140000\n')


#用pandas读取数据集
data_file = os.path.join('..','data','2_house_tiny.csv')
data=pd.read_csv(data_file)
print(data)

#用iloc将输入和输出分离
inputs,outputs = data.iloc[:,0:2],data.iloc[:,2]        #切片左闭右开

#用平均值替代NumRooms的NAN值
inputs = inputs.fillna(inputs.mean())
print(inputs)
#将类别值分成两个参数
inputs=pd.get_dummies(inputs,dummy_na=True)

#数据全部处理成数值类型后，即可转换成张量
X,y = torch.tensor(inputs.values),torch.tensor(outputs.values)
print(X,y)