import torch.nn as nn
import torch

def get_loss(loss, weight, **kwargs):
    if loss == 'CrossEntropy':
        weights = torch.Tensor([1/weight, 1])
        loss_fn = nn.CrossEntropyLoss(weight = weights)
    return loss_fn