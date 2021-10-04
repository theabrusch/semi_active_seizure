import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

def conv_size(input_size, kernel, padding, stride):
    '''
    Return output shape after convolution based on input
    size, kernel size, padding and stride. 
    '''
    if isinstance(kernel, int):
        kernel = (kernel, kernel)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride)
    h,w = input_size
    h_out = (h + 2*padding[0] - kernel[0])/stride[0] + 1
    w_out = (w + 2*padding[1] - kernel[1])/stride[1] + 1

    return (h_out, w_out)




class BaselineCNN(nn.Module):

    def __init__(self):
        super(BaselineCNN, self).__init__()
