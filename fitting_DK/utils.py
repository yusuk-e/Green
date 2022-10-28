
import math
import numpy as np
import os, torch, pickle, zipfile
import imageio, shutil
import scipy, scipy.misc, scipy.integrate

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import seaborn as sns
#import matplotlib.image as mpimg
#import matplotlib.animation as animation
solve_ivp = scipy.integrate.solve_ivp
import pdb
from t_io import standard_io as std_io
torch.set_default_dtype(torch.float64)

DPI = 200
FORMAT = 'pdf'
LINE_SEGMENTS = 10
ARROW_SCALE = 100
ARROW_WIDTH = 6e-3
LINE_WIDTH = 2
xmin = -3.2; xmax = 3.2; ymin = -3.2; ymax = 3.2


def choose_nonlinearity(name):
  nl = None
  if name == 'tanh':
    nl = torch.tanh
  elif name == 'relu':
    nl = torch.relu
  elif name == 'sigmoid':
    nl = torch.sigmoid
  elif name == 'softplus':
    nl = torch.nn.functional.softplus
  elif name == 'selu':
    nl = torch.nn.functional.selu
  elif name == 'elu':
    nl = torch.nn.functional.elu
  elif name == 'swish':
    nl = lambda x: x * torch.sigmoid(x)
  else:
    raise ValueError("nonlinearity not recognized")
  return nl
