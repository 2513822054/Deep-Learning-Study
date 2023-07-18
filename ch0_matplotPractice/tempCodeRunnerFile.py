# 在图中从位置(0,0)到位置(6,250)画一条线  
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