"""
  loss functions
"""
import torch
import logging
import numpy as np
import time
import torchvision


class LossMSE():
    """ mse loss """
    def __init__(self, params, model):
        self.params = params
        self.model = model

    def data(self, inputs, pred, target):
        if self.params.loss_style == 'mean':
            loss = torch.mean((target - pred)**2)
        elif self.params.loss_style == 'sum':
            loss = torch.sum((target - pred)**2)/pred.shape[0]
        return loss

    def bc(self, inputs, pred, targets):
        # currently no BC
        return torch.tensor(0.).to(self.params.device, dtype=torch.float32)

    def pde(self, inputs, pred, targets):
        # currently no PDE loss
        return torch.tensor(0.).to(self.params.device, dtype=torch.float32)

