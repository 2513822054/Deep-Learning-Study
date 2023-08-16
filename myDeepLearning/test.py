import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import myPlot
import requests
import subprocess
import os
import math

for i in range(4096):
    print(int(0.5*math.sin(math.pi*2*i/4096)/3.3*4096+2048),",",end="")
    # print(int())
# for i in range(1024):
#     print(int(0.5*i/1024*4096/3.3+2048),",",end="")
# for i in range(2048):
#     print(int(0.5*(1024-i)/1024*4096/3.3+2048),",",end="")
# for i in range(1024):
#     print(int(0.5*(-1024+i)/1024*4096/3.3+2048),",",end="")