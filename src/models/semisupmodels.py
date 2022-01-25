import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch
import numpy as np
from scipy.stats import truncnorm
import warnings
warnings.filterwarnings('ignore', 'Named tensors.*', )

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def get_trainable_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def conv_size(input_size, kernel, padding, stride=1):
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
    h_out = int((h + 2*padding[0] - kernel[0])/stride[0]) + 1
    w_out = int((w + 2*padding[1] - kernel[1])/stride[1]) + 1

    return h_out, w_out

class BaselineCNN(nn.Module):

    def __init__(self, input_shape, dropoutprob = 0.2, padding=True, 
                 glob_avg_pool = True, latent_size = 600, **kwargs):
        super(BaselineCNN, self).__init__() 
        ch_dim = input_shape[0]

        if padding:
            padding = [(0,5), (int(ch_dim/2),0), (5,5), (5,5)]
        else:
            padding = [(0,0), (int(ch_dim/2),0), (5,0), (5,0)]

        h, w = conv_size(input_shape, (1, 10), padding[0], stride = 1)
        h, w = conv_size((h,w), (ch_dim,1), padding[1], stride = 1)
        h, w = conv_size((h,w), (1,2), 0, stride=(1,2))
        h, w = conv_size((h,w), (10,10), padding[2], stride=1)
        h, w = conv_size((h,w), (1,2), 0, stride=(1,2))
        h, w = conv_size((h,w), (10,10), padding[3], stride=1)
        
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 20, 
                      kernel_size = (1,10), padding = padding[0]),
            nn.Conv2d(in_channels = 20, out_channels = 20, 
                      kernel_size=(ch_dim, 1), padding = padding[1]),
            nn.BatchNorm2d(20),
            nn.ELU(),
            nn.MaxPool2d(kernel_size = (1,2), stride = (1,2)),
            nn.Dropout(dropoutprob),
            nn.Conv2d(in_channels = 20, out_channels = 40, 
                      kernel_size = (10, 10), padding = padding[2]),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.MaxPool2d(kernel_size = (1,2), stride = (1,2)),
            nn.Dropout(dropoutprob),
            nn.Conv2d(in_channels = 40, out_channels = 80, 
                      kernel_size = (10, 10), padding = padding[3]),
            nn.BatchNorm2d(80),
            nn.Dropout(dropoutprob),
            nn.ELU()
        )

        if not glob_avg_pool:
            self.tolatent = nn.Sequential(nn.Flatten(),
                                             nn.Linear(in_features=h*w*80, out_features=latent_size),
                                             nn.ELU())
            self.fromlatent = nn.Linear(in_features = latent_size, out_features=h*w*80)
            self.classifier = nn.Linear(in_features = latent_size, out_features=2)
                                             
        else:
            self.tolatent = nn.Sequential(nn.AvgPool2d(kernel_size=(h,w),
                                                          stride = (h,w)),
                                             nn.Flatten(),
                                             nn.ELU())
            self.fromlatent = nn.Linear(in_features = 80, out_features=h*w*80)
            self.classifier = nn.Linear(in_features = 80, out_features=2)

        self.transconvblock = nn.Sequential(
            
        )
    def forward(self, x, training=True):
        x = self.convblock(x.unsqueeze(1))
        latent = self.tolatent(x)
        x_tilde = self.fromlatent(latent)
        out = self.classifier(latent)
        out = F.softmax(x, dim = 1)
        return out
