from os import error
import torch.optim as optim

def get_optim(model, optim_kwargs):
    if optim_kwargs['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr = optim_kwargs['lr'],
                               weight_decay=optim_kwargs['weight_decay'])
    elif optim_kwargs['optimizer'] == 'RMSprop':
        if optim_kwargs['model'] == 'AttBiLSTM':
            params = [{'params': model.att.parameters()},
                      {'params': model.fc1.parameters()},
                      {'params': model.fc2.parameters()},
                      {'params': model.lstm.parameters(), 'weight_decay': 0}]
            optimizer = optim.RMSprop(params, lr = optim_kwargs['lr'], 
                                      weight_decay=optim_kwargs['weight_decay'])
        else:
            optimizer = optim.RMSprop(model.parameters(), lr = optim_kwargs['lr'], 
                                    weight_decay=optim_kwargs['weight_decay'])
    else:
        raise ValueError('Available optimizers are Adam and RMSprop.')
    
    if optim_kwargs['scheduler'] == 'MultistepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, optim_kwargs['milestones'])
    else:
        scheduler = None
    
    return optimizer, scheduler
    


