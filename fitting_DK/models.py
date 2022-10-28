import sys
import pdb
import torch
import torch.nn as nn
import numpy as np
import math
import time
from scipy.spatial.distance import cdist
torch.set_default_dtype(torch.float64)

from utils import choose_nonlinearity


class nn(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity):
    super(nn, self).__init__()
    self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
    self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear4 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear5 = torch.nn.Linear(hidden_dim, output_dim, bias=None)

    for l in [self.linear1, self.linear2, self.linear3, self.linear4, self.linear5]:
      torch.nn.init.orthogonal_(l.weight)
      #torch.nn.init.normal_(l.weight, mean=0.0, std=1.0) 
      
    self.nonlinearity = choose_nonlinearity(nonlinearity)

  def forward(self, x):
    x1 = x.unsqueeze(1).expand(-1,len(x)).reshape(-1,1)
    x2 = x.expand(len(x),-1).reshape(-1,1)
    x12 = torch.hstack([x1,x2])

    h = self.nonlinearity( self.linear1(x12) )
    h = self.nonlinearity( self.linear2(h) )
    h = self.nonlinearity( self.linear3(h) )
    h = self.nonlinearity( self.linear4(h) )
    h = self.linear5(h)
    return h.reshape(len(x),len(x))

  def loss(self, xs, target_K):
    pred_K = self.forward(xs)
    return ((target_K - pred_K)**2).sum()/2.

class nn_s_invar(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity):
    super(nn_s_invar, self).__init__()
    self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
    self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear4 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear5 = torch.nn.Linear(hidden_dim, output_dim, bias=None)

    for l in [self.linear1, self.linear2, self.linear3, self.linear4, self.linear5]:
    #for l in [self.linear1, self.linear2, self.linear3]:
      torch.nn.init.orthogonal_(l.weight)
      #torch.nn.init.normal_(l.weight, mean=0.0, std=1.0) 
      
    self.nonlinearity = choose_nonlinearity(nonlinearity)

  def forward(self, x):
    x1 = x.unsqueeze(1).expand(-1,len(x)).reshape(-1,1)
    x2 = x.expand(len(x),-1).reshape(-1,1)
    x12 = torch.hstack([x1,x2])
    x21 = torch.hstack([x2,x1])

    h = self.nonlinearity( self.linear1(x12) )
    h = self.nonlinearity( self.linear2(h) )
    h = self.nonlinearity( self.linear3(h) )
    h = self.nonlinearity( self.linear4(h) )
    h12 = self.linear5(h)

    h = self.nonlinearity( self.linear1(x21) )
    h = self.nonlinearity( self.linear2(h) )
    h = self.nonlinearity( self.linear3(h) )
    h = self.nonlinearity( self.linear4(h) )
    h21 = self.linear5(h)

    return (h12 + h21).reshape(len(x),len(x))

  def loss(self, xs, target_K):
    pred_K = self.forward(xs)
    return ((target_K - pred_K)**2).sum()/2.


