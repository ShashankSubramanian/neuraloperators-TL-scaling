import os, sys
import logging
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

def set_scheduler(args, optimizer):
    """ set the lr scheduler """
    if args.scheduler == 'reducelr':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.patience, verbose=True, min_lr=1e-3*1e-5, factor=0.2)
    elif args.scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_cosine_lr_epochs)
    else:
        scheduler = None
    return scheduler

def set_optimizer(args, net):
    """ set the optimizer """
    if args.optimizer == "adam":
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)    
    return optimizer

