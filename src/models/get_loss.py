import torch.nn as nn
import torch

def get_loss(loss, weight, **kwargs):
    if loss == 'CrossEntropy':
        if weight is not None:
            weights = torch.Tensor([1/weight, 1])
            loss_fn = nn.CrossEntropyLoss(weight = weights)
        else:
            loss_fn = nn.CrossEntropyLoss()
    return loss_fn
