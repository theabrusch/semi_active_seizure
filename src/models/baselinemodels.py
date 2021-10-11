import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore', 'Named tensors.*', )


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

    def __init__(self, input_shape, **kwargs):
        super(BaselineCNN, self).__init__() 

        h, w = conv_size(input_shape, (1,2), 0, stride=(1,2))
        h, w = conv_size((h,w), (1,2), 0, stride=(1,2))
        
        ch_dim = [input_shape[0] if input_shape[0]%2 != 0 else input_shape[0]-1][0]

        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 20, 
                      kernel_size = (1,11), padding = 'same'),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.BatchNorm2d(20),
            nn.Conv2d(in_channels = 20, out_channels = 20, 
                      kernel_size=(ch_dim, 21), padding = 'same'),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.BatchNorm2d(20),
            nn.MaxPool2d(kernel_size = (1,2), stride = (1,2)),
            nn.Conv2d(in_channels = 20, out_channels = 40, 
                      kernel_size = (11, 21), padding = 'same'),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.BatchNorm2d(40),
            nn.MaxPool2d(kernel_size = (1,2), stride = (1,2)),
            nn.Conv2d(in_channels = 40, out_channels=80, 
                      kernel_size = (11, 11), padding = 'same')
        )
        self.dropout = nn.Dropout(0.4)
        self.GlobAvgPool = nn.AvgPool2d(kernel_size=(h,w), stride =(h,w))
        self.fc = nn.Linear(in_features=80, out_features=2)
    
    def forward(self,x):
        x = self.convblock(x.unsqueeze(1))
        x = self.GlobAvgPool(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = x.squeeze(3).squeeze(2)
        x = self.fc(x)
        out = F.softmax(x, dim = 1)
        return out