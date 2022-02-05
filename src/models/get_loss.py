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

def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    x = torch.flatten(x, start_dim = 1)
    y = torch.flatten(y, start_dim = 1)
    dim = x.size(1)
    x = x.unsqueeze(1)  # (x_size, 1, dim)
    y = y.unsqueeze(0)  # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)
    return torch.exp(-kernel_input)  # (x_size, y_size)


def mmd_loss(x, y):
	x_kernel = compute_kernel(x, x)
	y_kernel = compute_kernel(y, y)
	xy_kernel = compute_kernel(x, y)
	mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
	return mmd

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

class Buffer(nn.Module):
    def __init__(self, last_k):
        super(Buffer, self).__init__()
        self.last_k = last_k
        self.buffer = torch.Tensor([])

    def forward(self, features):
        self.buffer = torch.cat([self.buffer, features])[-self.last_k:].detach()
        return self.buffer

class SemiSupervisedTransfer(nn.Module):
    def __init__(self, 
                 classification_loss,
                 lambda_cons = 1,
                 lambda_rep = 1,
                 get_last_k = 20,
                 epsilon_cons = 0.7,
                 epsilon_rep = 0.5):
        '''
        Class for implementing a transfer loss inspired by 
        'Adaptive Consistency Regularization for Semi-Supervised Transfer Learning'
        '''
        super(SemiSupervisedTransfer, self).__init__()
        self.classification_loss = classification_loss
        self.epsilon_cons = epsilon_cons*torch.log(torch.as_tensor(2))
        self.epsilon_rep = epsilon_rep*torch.log(torch.as_tensor(2))
        self.lambda_cons = lambda_cons
        self.lambda_rep = lambda_rep
        
        # labeled and unlabeled buffer
        self.get_last_k = get_last_k
        self.labeled_buffer = Buffer(get_last_k)
        self.unlabeled_buffer = Buffer(get_last_k)        

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
        loss = self.classification_loss(out_target_labeled, y_true)
        if self.lambda_cons > 0: # adaptive consistency loss
            # choose examples that were probably correctly classified 
            # by the source model 
            out_source = torch.cat((out_source_labeled, out_source_unlabeled), dim = 0)
            entropy = -torch.sum(out_source*torch.log(out_source), dim = 1)
            mask = (entropy <= self.epsilon_cons)
            
            features_target = torch.cat((features_target_labeled, features_target_unlabeled), dim = 0)
            features_source = torch.cat((features_source_labeled, features_source_unlabeled), dim = 0)

            akc = F.mse_loss(features_target, features_source, reduction = 'none')[mask,...].mean()
            loss += self.lambda_cons*akc

        if self.lambda_rep > 0: # adaptive representation consistency
            # get features using entropy
            labeled_entropy = -torch.sum(out_target_labeled*torch.log(out_target_labeled), dim = 1)
            unlabeled_entropy = -torch.sum(out_target_unlabeled*torch.log(out_target_unlabeled), dim = 1)
            labeled_mask = (labeled_entropy <= self.epsilon_rep)
            unlabeled_mask = (unlabeled_entropy <= self.epsilon_rep)

            labeled_target_features = features_target_labeled[labeled_mask,...]
            unlabeled_target_features = features_target_unlabeled[unlabeled_mask,...]
            
            # update buffers
            labeled_target_features = self.labeled_buffer(labeled_target_features)
            unlabeled_target_features = self.unlabeled_buffer(unlabeled_target_features)

            if len(labeled_target_features) > 20:
                arc = mmd_loss(labeled_target_features, unlabeled_target_features)
                loss += self.lambda_rep*arc

        return loss