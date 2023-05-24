'''
  domain classes
'''
import torch
import random
import numpy as np
from utils.misc_utils import normalize, softmax, show
import matplotlib.pyplot as plt

class DomainXY():
    """ 
        Creates a uniform grid of 2D spatial points
    """
    def __init__(self, params):
        self.params = params
        dx = params.Lx / params.nx
        dy = params.Ly / params.ny
        self.dx = dx
        self.dy = dy
        self.x = np.arange(0, params.Lx, dx)
        self.y = np.arange(0, params.Ly, dy)
        x_g, y_g = np.meshgrid(self.x, self.y)
        self.x_g, self.y_g = x_g, y_g
        self.grid = np.column_stack((x_g.flatten(), y_g.flatten()))

