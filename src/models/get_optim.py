import torch.optim as optim

def get_optim(model_params, optim_kwargs):
    if optim_kwargs['optimizer'] == 'Adam':
        optimizer = optim.Adam(model_params, lr = optim_kwargs['lr'],
                               weight_decay=optim_kwargs['weight_decay'])
    
    if optim_kwargs['scheduler'] == 'MultistepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, optim_kwargs['milestones'])
    else:
        scheduler = None
    
    return optimizer, scheduler
    


