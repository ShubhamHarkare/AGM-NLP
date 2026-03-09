import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx,x,lambda_):
        #DESC: This function is responsible for the forward pass in the DANN model
        #DESC: This will take in a feature tensor and return the same feature tensor

        ctx.save_for_backward(torch.tensor(lambda_)) #! This line helps to preserve the value for the backward pass
        return x
        

    @staticmethod
    def backward(ctx,grad_output):
        lambda_, = ctx.saved_tensors
        return -lambda_ * grad_output, None


        
