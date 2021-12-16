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

    def __init__(self, input_shape, dropoutprob = 0.2, padding=True, **kwargs):
        super(BaselineCNN, self).__init__() 

        ch_dim = input_shape[0]

        if padding:
            padding = [(0,5), (int(ch_dim/2),0), (5,5), (5,5)]
        else:
            padding = [(0,0), (int(ch_dim/2),0), (0,0), (0,0)]

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
            nn.MaxPool2d(kernel_size = (1,2), stride = (1,2)),
            nn.ELU(),
            nn.Dropout(dropoutprob),
            nn.BatchNorm2d(20),
            nn.Conv2d(in_channels = 20, out_channels = 40, 
                      kernel_size = (10, 10), padding = padding[2]),
            nn.MaxPool2d(kernel_size = (1,2), stride = (1,2)),
            nn.ELU(),
            nn.Dropout(dropoutprob),
            nn.BatchNorm2d(40),
            nn.Conv2d(in_channels = 40, out_channels=80, 
                      kernel_size = (10, 10), padding = padding[3])
        )
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropoutprob)
        self.fc = nn.Linear(in_features=h*w*80, out_features=2)
    
    def forward(self,x, training=True):
        x = self.convblock(x.unsqueeze(1))
        x = self.flatten(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.fc(x)
        out = F.softmax(x, dim = 1)
        return out
    
class AttentionModule(nn.Module):
    def __init__(self, input_shape):
        super(AttentionModule, self).__init__()
        self.time_steps, self.channels = input_shape[1], input_shape[0]
        # attention weights
        self.fc = nn.Linear(in_features = self.channels, 
                            out_features = self.channels)
        # draw weights from truncated normal
        truncated_normal_init = truncnorm(0, 10, scale = 0.1)
        self.fc.weight = Parameter(torch.Tensor(truncated_normal_init.rvs(size = self.fc.weight.shape)).float())
    
    def forward(self, x):
        a = self.fc(x)
        a = F.softmax(a, dim = 2)
        # apply same attention across timesteps
        a = torch.mean(a, axis = 1).unsqueeze(1)
        x = x*a
        return x

class AttentionBiLSTM(nn.Module):
    def __init__(self, input_shape, lstm_units, dense_units, **kwargs):
        super(AttentionBiLSTM, self).__init__()
        self.time_steps, self.channels = input_shape[1], input_shape[0]
        # attention layer
        self.att = AttentionModule(input_shape)
        # lstm
        self.lstm = nn.LSTM(input_size = self.channels,
                            hidden_size = lstm_units,
                            batch_first = True,
                            bidirectional = True)
        self.fc1 = nn.Linear(in_features=lstm_units*2, 
                             out_features=dense_units)
        self.globAvg = nn.AvgPool2d(kernel_size=(self.time_steps, 1),
                                    stride = (self.time_steps, 1))
        self.fc2 = nn.Linear(in_features = dense_units, 
                             out_features = 2)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc2.bias.data.fill_(0.1)

    def forward(self, x, training = False):
        x = torch.permute(x, (0,2,1))
        x = self.att(x)
        x,_ = self.lstm(x)
        x = self.fc1(x)
        x = self.globAvg(x.unsqueeze(1)).squeeze(2).squeeze(1)
        x = self.fc2(x)
        return F.softmax(x, dim = 1)
