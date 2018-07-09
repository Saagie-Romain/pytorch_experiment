import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data
import math, copy, time
from torch.autograd import Variable



#### Device choice
#num_device = 0
#device = torch.device("cuda:" + str(num_device) if torch.cuda.is_available() else "cpu")
#torch.cuda.set_device(num_device

print("coucou")