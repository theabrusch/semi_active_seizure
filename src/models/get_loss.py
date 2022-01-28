import torch.nn as nn
import torch
import torch.nn.functional as F

def get_loss(loss, weight, **kwargs):
    if loss == 'CrossEntropy':
        if weight is not None:
            weights = torch.Tensor([1/weight, 1])
            loss_fn = nn.CrossEntropyLoss(weight = weights)
        else:
            loss_fn = nn.CrossEntropyLoss()
    return loss_fn


class TransferLoss():
    def __init__(self, 
                 classification_loss,
                 lambda_cons = 1,
                 use_entropy = False, 
                 epsilon = None):
        '''
        Class for implementing a transfer loss inspired by 
        'Adaptive Consistency Regularization for Semi-Supervised Transfer Learning'
        '''
        self.classification_loss = classification_loss
        self.use_entropy = use_entropy
        self.epsilon = epsilon
        self.lambda_cons = lambda_cons

    def __call__(self, 
                out_target, 
                features_target, 
                out_source,
                features_source,
                y_true):

        if not self.use_entropy:
            # choose examples that were correctly classified 
            # by the source model 
            source_pred = torch.argmax(out_source, dim = 1)
            mask = (y_true.squeeze() == source_pred)
        
        mse = F.mse_loss(features_target, features_source, reduction = 'none')[mask,...].mean()
        class_loss = self.classification_loss(out_target, y_true.squeeze())

        return class_loss + self.lambda_cons*mse