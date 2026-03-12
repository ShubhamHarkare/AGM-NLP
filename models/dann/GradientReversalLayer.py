import torch
import torch.nn as nn
# from dann import GradientReversalFunction
from GradientReversalFunction import GradientReversalFunction

class GradientReversalLayer(nn.Module):
    def __init__(self,lambda_):
        super().__init__()
        self.lambda_ = lambda_


    def forward(self,x):
        return GradientReversalFunction.apply(x,self.lambda_)