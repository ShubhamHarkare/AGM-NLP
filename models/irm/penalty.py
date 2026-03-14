
#DESC: This file contains the code to calucate the penalty term for Invariant Risk Minimization neural network
from venv import create

import torch
import torch.nn as nn





def compute_irm_penalty(logits,labels,criterion):
    '''
    This function is responsible for calucalating the penalty term for IRM.
    '''
    w = torch.tensor(1.0,requires_grad=True)
    loss = criterion(w * logits,labels)
    grad = torch.autograd.grad(loss,w,create_graph=True)[0]

    return grad ** 2