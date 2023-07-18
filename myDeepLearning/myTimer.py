import time
import numpy as np
class Timer:
    def __init__(self):
        self.times = []
        self.start()
    def start(self):
        '''启动定时器'''
        self.tick = time.time()
    def stop(self):
        '''停止计时器并将时间记录在列表中'''
        self.times.append(time.time()-self.tick)
        return self.times[-1]
    def avg(self):
        '''返回平均时间'''
        return sum(self.times)/len(self.times)
    def sum(self):
        '''返回时间总和'''
        return sum(self.times)
    def cumsum(self):
        return np.array(self.times).cumsum().tolist()
