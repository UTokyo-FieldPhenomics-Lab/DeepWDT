import torch
from torch import optim


def build_optimizer(solver, model, base_lr=0.0, resume=None):

    if solver.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), 
            lr=base_lr,
            momentum=solver.momentum,
            weight_decay=solver.weight_decay)

    elif solver.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(), 
            lr=base_lr,
            weight_decay=solver.weight_decay)
                                
    elif solver.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=base_lr,
            weight_decay=solver.weight_decay)
          
    start_epoch = 0

    if resume:
        print('keep training: ', resume)
        checkpoint = torch.load(resume)
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("optimizer")
        optimizer.load_state_dict(checkpoint_state_dict)
        start_epoch = checkpoint.pop("epoch")
                        
                                
    return optimizer, start_epoch
