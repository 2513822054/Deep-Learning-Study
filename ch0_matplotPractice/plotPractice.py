import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# # 在图中从位置(0,0)到位置(6,250)画一条线  
# xpoints = np.array([0, 6])  
# ypoints = np.array([0, 250])  
# plt.plot(xpoints, ypoints)  
# plt.show()

# # 不指定x轴的点，默认为0到1平均分
# ypoints = np.array([0, 250])
# plt.plot(ypoints)  
# plt.show()

# #仅绘制标记点，可以使用快捷字符串符号参数 ‘o’ ,这意味着“环”
# xpoints = np.array([0, 6])  
# ypoints = np.array([0, 250])  
# plt.plot(xpoints, ypoints,'o')  
# plt.show()

# #可以根据需要绘制任意数量的点，只需确保两个轴上的点数相同即可   连接线会按顺序连起来
# xpoints = np.array([33, 7, 6, 13])  
# ypoints = np.array([3, 23, 88, 42])  
# plt.plot(xpoints, ypoints)  
# plt.show()

#关键字：marker，用指定的标记强调每个点  color用来设定颜色  markersize标记点大小   linestyle线条样式  linewidth线条宽度
xpoints = np.array([1, 3, 5, 7])  
ypoints = np.array([3, 23, 88, 42])  
ypoints2 = np.array([78, 13, 44, 99])  
#使用subplots()函数来显示多张图  subplots(几行，几列，第几张子图)  在绘制plot函数之前使用
plt.subplot(1,2,1)
# matplotlib.rcParams['font.sans-serif'] = ['KaiTi']      #设置xy轴的label前需要设置字体
# plt.xlabel('品质')
# plt.ylabel('价格')
#也可以使用不同的字体
font1 = {'family': 'KaiTi', 'color': 'red', 'size': 20}  
font2 = {'family': 'KaiTi', 'color': 'darkred', 'size': 15} 
plt.title('我是标题', fontdict=font1,loc='center')              #设置标题   关键字 loc，标题位置
plt.xlabel('时间节点', fontdict=font2)  
plt.ylabel('收入', fontdict=font2)
plt.grid(axis='both',color='grey',linestyle='dashed',linewidth='1')           #添加网格线  axis，横向还是纵向网格线  color，颜色 
plt.plot(xpoints, ypoints, marker='.',color='purple',markersize='20',linestyle='dashed',linewidth='7')  
plt.plot(xpoints, ypoints2)                 #多个plot画多条线
plt.subplot(1,2,2)
plt.plot(xpoints, ypoints2, marker='*',color='purple',markersize='10',linestyle='-.',linewidth='3')  
plt.suptitle('我是标题2', fontdict=font1)             #为整个大表添加标题
plt.show()
