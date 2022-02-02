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


class TransferLoss(nn.Module):
    def __init__(self, 
                 classification_loss,
                 lambda_cons = 1,
                 use_entropy = False, 
                 epsilon = None):
        '''
        Class for implementing a transfer loss inspired by 
        'Adaptive Consistency Regularization for Semi-Supervised Transfer Learning'
        '''
        super(TransferLoss, self).__init__()
        self.classification_loss = classification_loss
        self.use_entropy = use_entropy
        self.epsilon = epsilon
        self.lambda_cons = lambda_cons

    def forward(self, 
                out_target, 
                features_target, 
                out_source,
                features_source,
                y_true):

        if not self.use_entropy and self.lambda_cons > 0:
            # choose examples that were correctly classified 
            # by the source model 
            source_pred = torch.argmax(out_source, dim = 1)
            mask = (y_true == source_pred)
        
            mse = F.mse_loss(features_target, features_source, reduction = 'none')[mask,...].mean()
        class_loss = self.classification_loss(out_target, y_true)
        if self.lambda_cons > 0:
            return class_loss + self.lambda_cons*mse
        else:
            return class_loss



class SemiSupervisedTransfer(nn.Module):
    def __init__(self, 
                 classification_loss,
                 lambda_cons = 1,
                 lambda_rep = 1,
                 epsilon_cons = 0,
                 epsilon_rep = 0):
        '''
        Class for implementing a transfer loss inspired by 
        'Adaptive Consistency Regularization for Semi-Supervised Transfer Learning'
        '''
        super(TransferLoss, self).__init__()
        self.classification_loss = classification_loss
        self.epsilon_cons = epsilon_cons
        self.epsilon_rep = epsilon_rep
        self.lambda_cons = lambda_cons
        self.lambda_rep = lambda_rep

    def forward(self, 
                out_target_labeled, 
                features_target_labeled, 
                features_target_unlabeled,
                out_target_unlabeled,
                out_source_labeled,
                features_source_labeled,
                features_source_unlabeled,
                out_source_unlabeled,
                y_true):
        '''
        Takes the output of the source and the target model and computes the 
        AKC-ARC loss.
        '''
        if self.lambda_cons > 0:
            # choose examples that were probably correctly classified 
            # by the source model 
            out_source = torch.cat((out_source_labeled, out_source_unlabeled), dim = 0)
            entropy = -torch.sum(out_source*torch.log(out_source), dim = 1)
            mask = (entropy <= self.lambda_cons)
            
            features_target = torch.cat((features_target_labeled, features_target_unlabeled), dim = 0)
            features_source = torch.cat((features_source_labeled, features_source_unlabeled), dim = 0)

            akc = F.mse_loss(features_target, features_source, reduction = 'none')[mask,...].mean()

        class_loss = self.classification_loss(out_target, y_true)
        if self.lambda_cons > 0:
            return class_loss + self.lambda_cons*akc
        else:
            return class_loss