import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import myPlot

use_gpu = torch.cuda.is_available()
print(use_gpu)
print(torch.version.cuda)