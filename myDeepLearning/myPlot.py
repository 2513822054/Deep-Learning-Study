import torch
from torch.distributions import multinomial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline
from IPython import display

font_suptitle_dic={'family': 'FangSong', 'color': 'Black', 'size': 20,'weight':'normal','style':'oblique'}
font_title_dic={'family': 'FangSong', 'color': 'Black', 'size': 20,'weight':'normal','style':'oblique'}
font_xylabel_dic={'family': 'FangSong', 'color': 'Black', 'size': 16,'weight':'light','style':'italic'}

class myFigure():
    '''
    构造函数
    fabric:子图结构,[n,m]表示有n行、列子图
    figsize:视图大小, 视图宽度为figsize[0]*dpi,高度为figsize[1]*dpi
    dpi:用来控制视图大小
    suptitile:总标题
    font_suptitle:总标题字体样式
    '''
    def __init__(self,
                fabric=[1,1],
                figsize=None,
                dpi=100,
                suptitle=None,
                font_suptitle = font_suptitle_dic,
                ):
        self.fabric = fabric
        if figsize==None:
            self.figsize=[fabric[1]*5,fabric[0]*5]
        else:
            self.figsize = figsize
        self.dpi = dpi
        self.fig = plt.figure(figsize=self.figsize,dpi=self.dpi)
        self.ax = []
        self.sub_num = fabric[0]*fabric[1]
        for i in range(self.sub_num):
            self.ax.append(self.fig.add_subplot(fabric[0],fabric[1],i+1))
        self.suptitle = suptitle
        if suptitle!=None:
            self.fig.suptitle(suptitle,fontdict=font_suptitle)
    
    '''
    添加图表
    inputy:y轴输入
    inputx:X轴数据,如果没输入,则以1~len代替
    index:子图序号,范围从1~(fabric[0]*fabric[1]),也可以是,[第n行,第m列]子图,其中n和m都布恩那个超过子图行数和列数
    dim:数据沿哪一根轴展开如果为0,则第i根曲线的数据依次是data[i][1] data[i][2]...;如果为1则第i根曲线的数据依次是data[1][i] data[2][i]
    title:图表标题
    xLabel,yLabel:x y 轴标签
    xGrid,yGrid:是否显示x轴y轴网格
    font_title:标题字体
    font_xylabel:xy标签字体
    xlim ylim:x和y的显示范围
    xscale yscale:xy轴的缩放  “linear” “log” “symlog” “logit”
    '''
    def add_graph(
                self,
                inputy,
                inputx = None,
                index = 1,
                dim=0,
                scatter = False,
                title="",
                xlabel="",ylabel="",
                xGrid=True,yGrid=True,
                font_title = font_title_dic,
                font_xylabel = font_xylabel_dic,
                xlim=None,ylim=None,
                legend=None,
                xscale='linear',yscale='linear',
                ):
        '''Error judge'''
        if type(index) == list:
            if index[0]>self.fabric[0] or index[0]<1:
                raise ValueError("index[0] must be between 1 and %d(the number of rows of subgraph); got %d"%self.fabric[0],index[0])
            elif index[1]>self.fabric[1] or index[1]<1:
                raise ValueError("index[1] must be between 1 and %d(the number of columns of subgraph); got %d"%self.fabric[1],index[1])
            index = (index[0]-1)*self.fabric[1] + index[1]
        if index>self.sub_num:
            raise ValueError("index must be between 1 and %d(the number of subgraphs); got %d"%self.sub_num,index)
        index-=1
        if not(isinstance(inputy, np.ndarray) or isinstance(inputy, torch.Tensor) or isinstance(inputy,list)):
            raise TypeError("TypeError: inputy must be a ndarry or torch.tensor or list; got: %s"%type(inputy))
        if isinstance(inputy, torch.Tensor):
            inputy = inputy.numpy()
        elif isinstance(inputy, list):
            inputy = np.array(inputy)
        if dim==1:
            inputy = inputy.T
        shape = inputy.shape
        if len(shape)<2:
            inputy = inputy.reshape((shape[0],1))
            shape = inputy.shape
        if inputx != None:
            if not(isinstance(inputx, np.ndarray) or isinstance(inputx, torch.Tensor) or isinstance(inputx,list)):
                raise TypeError("TypeError: inputx must be a ndarry or torch.tensor or list; got: %s"%type(inputx))
            if isinstance(inputx, torch.Tensor):
                inputx = inputx.numpy()
            elif isinstance(inputx, list):
                inputx = np.array(inputx)
            x = inputx.reshape(inputx.size)
        else:
            x = np.arange(1,shape[0]+1,1)
        '''Error judge end'''
        for i in range(shape[1]):
            if scatter:
                self.ax[index].scatter(x, inputy[:,i:i+1])
            else:
                self.ax[index].plot(x, inputy[:,i:i+1])
        if xGrid and yGrid:
            self.ax[index].grid()
        elif xGrid:
            self.ax[index].grid(axis='x')
        elif yGrid:
            self.ax[index].grid(axis='y')
        #self.ax[index].set(title=title,xlim=xlim,ylim=ylim)
        if title != '':
            self.ax[index].set_title(title, fontdict=font_title,loc='center')
        if xlabel != '':
            self.ax[index].set_xlabel(xlabel, fontdict=font_xylabel,labelpad=0)
        if ylabel != '':
            self.ax[index].set_ylabel(ylabel, fontdict=font_xylabel,labelpad=0)
        if xlim:
            self.ax[index].set_xlim(xlim)
        if ylim:
            self.ax[index].set_ylim(ylim)
        if legend != None:
            self.ax[index].legend(legend)
        self.ax[index].set_yscale(yscale)
        self.ax[index].set_xscale(xscale)

    def show(self):
        plt.show()

#显示多跟曲线，传进来的参数dim=0表示data的0轴是曲线图的x轴,   inputx表示输入x轴参数，否则以0~n-1为参数
'''
显示多跟曲线
data:输入的曲线数据
inputx:X轴数据,如果没输入,则以0~len-1代替
dim:数据沿哪一根轴展开如果为0,则第i根曲线的数据依次是data[i][1] data[i][2]...;如果为1则第i根曲线的数据依次是data[1][i] data[2][i]
title:图表标题
xLabel,yLabel:x y 轴标签
xGrid,yGrid:是否显示x轴y轴网格
font_title:标题字体
font_xylabel:xy标签字体
xlim ylim:x和y的显示范围
'''
def show_linears(data,
                 inputx = None,
                 dim=0,
                 scatter = False,
                 title="",
                 xLabel="",yLabel="",
                 xGrid=True,yGrid=True,
                 font_title = font_title_dic,
                 font_xylabel = font_xylabel_dic,
                 xlim=None,ylim=None,
                 legend=None,
                 ):
    '''Error judge'''
    if not(isinstance(data, np.ndarray) or isinstance(data, torch.Tensor)):
        raise TypeError("TypeError: data must be a ndarry or torch.tensor; got: %s"%type(data))
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    shape = data.shape
    if inputx != None:
        if not(isinstance(inputx, np.ndarray) or isinstance(inputx, torch.Tensor)):
            raise TypeError("TypeError: inputx must be a ndarry or torch.tensor; got: %s"%type(inputx))
        if isinstance(inputx, torch.Tensor):
            inputx = inputx.numpy()
        x = inputx.reshape(inputx.size)
    else:
        x = np.arange(0,shape[0],1)
    '''Error judge end'''
    # if inputx != None:
    #     x = inputx.reshape(inputx.size())
    # else:
    #     x = np.arange(0,shape[0],1)
    for i in range(shape[1]):
        if scatter:
            plt.scatter(x, data[:,i:i+1])
        else:
            plt.plot(x, data[:,i:i+1])
    if xGrid and yGrid:
        plt.grid()
    elif xGrid:
        plt.grid(axis='x')
    elif yGrid:
        plt.grid(axis='y')
    if title != '':
        plt.title(title, fontdict=font_title,loc='center')
    if xLabel != '':
        plt.xlabel(xLabel, fontdict=font_xylabel)
    if yLabel != '':
        plt.ylabel(yLabel, fontdict=font_xylabel)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    if legend != None:
        plt.legend(legend)
    plt.show()

def show_scatters(datax,datay,dim=0,title="",xLabel="",yLabel="",xGrid=True,yGrid=True,font_title = font_title_dic,font_xylabel = font_xylabel_dic):
    '''Error judge'''
    if not(isinstance(datax, np.ndarray) or isinstance(datax, torch.Tensor)):
        raise TypeError("TypeError: datax must be a ndarry or torch.tensor; got: %s"%type(datax))
    if isinstance(datax, torch.Tensor):
        datax = datax.numpy()
    if not(isinstance(datay, np.ndarray) or isinstance(datay, torch.Tensor)):
        raise TypeError("TypeError: datay must be a ndarry or torch.tensor; got: %s"%type(datay))
    if isinstance(datay, torch.Tensor):
        datay = datay.numpy()
    '''Error judge end'''
    datax = datax.reshape([datax.size,])
    datay = datay.reshape((datay.size,))
    plt.scatter(datax,datay)
    
    if xGrid and yGrid:
        plt.grid()
    elif xGrid:
        plt.grid(axis='x')
    elif yGrid:
        plt.grid(axis='y')
    if title != '':
        plt.title(title, fontdict=font_title,loc='center',)
    if xLabel != '':
        plt.xlabel(xLabel, fontdict=font_xylabel)
    if yLabel != '':
        plt.ylabel(yLabel, fontdict=font_xylabel)
    plt.show()

def show_scatters3D(datax,datay,dataz,dim=0,title="",xLabel="x",yLabel="y",zLabel="z",xGrid=True,yGrid=True,font_title = font_title_dic,font_xylabel = font_xylabel_dic):
    '''Error judge'''
    if not(isinstance(datax, np.ndarray) or isinstance(datax, torch.Tensor)):
        raise TypeError("TypeError: datax must be a ndarry or torch.tensor; got: %s"%type(datax))
    if isinstance(datax, torch.Tensor):
        datax = datax.numpy()
    if not(isinstance(datay, np.ndarray) or isinstance(datay, torch.Tensor)):
        raise TypeError("TypeError: datay must be a ndarry or torch.tensor; got: %s"%type(datay))
    if isinstance(datay, torch.Tensor):
        datay = datay.numpy()
    if not(isinstance(dataz, np.ndarray) or isinstance(dataz, torch.Tensor)):
        raise TypeError("TypeError: dataz must be a ndarry or torch.tensor; got: %s"%type(dataz))
    if isinstance(dataz, torch.Tensor):
        dataz = dataz.numpy()
    '''Error judge end'''
    datax = datax.reshape([datax.size,])
    datay = datay.reshape((datay.size,))
    dataz = dataz.reshape([dataz.size,])
    plt.figure(figsize = (10, 7))  
    ax = plt.axes(projection ="3d")
    ax.scatter(datax,datay,dataz)
    ax.set_xlabel(xLabel, fontdict=font_xylabel)
    ax.set_ylabel(yLabel, fontdict=font_xylabel)
    ax.set_zlabel(zLabel, fontdict=font_xylabel)
    plt.show()

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    plt.show()
    return axes

def use_svg_display():
    '''使用svg格式在Jupyter中显示绘图'''
    backend_inline.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5,2.5)):
    '''设置matplotlib的图表大小'''
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib.

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

class Animator:
    """For plotting data in animation."""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        """Defined in :numref:`sec_softmax_scratch`"""
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        plt.draw()
        plt.pause(0.001)        
        display.display(self.fig)
        display.clear_output(wait=True)