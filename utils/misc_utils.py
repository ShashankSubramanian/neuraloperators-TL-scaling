"""
  misc utils
"""

import numpy as np
import scipy as sc
import scipy.ndimage as nd
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
from matplotlib import cm
from torch.nn import functional as F

def fft_coef(n, is_torch=False):
    if not is_torch:
        ikx_pos = 1j * np.arange(0, n//2+1, 1)
        ikx_neg = 1j * np.arange(-n//2+1, 0, 1)
        ikx = np.concatenate((ikx_pos, ikx_neg))
    else:
        ikx_pos = 1j * torch.arange(0, n/2+1, 1)
        ikx_neg = 1j * torch.arange(-n/2+1, 0, 1)
        ikx = torch.cat((ikx_pos, ikx_neg))
    return ikx

def get_fft_coef(u, is_torch=False):
    nx, ny = u.shape
    ikx = fft_coef(nx, is_torch).reshape(1,nx)
    iky = fft_coef(ny, is_torch).reshape(ny,1)
    if not is_torch:
        ikx = np.repeat(ikx, ny, axis=0)
        iky = np.repeat(iky, nx, axis=1)
    else:
        # need to check if this is the same..
        ikx = torch.repeat_interleave(ikx, ny, dim=0)
        iky = torch.repeat_interleave(iky, nx, dim=1)
    return ikx, iky

def laplacian(u, params):
    ikx, iky = get_fft_coef(u)
    ikx2 = ikx**2
    iky2 = iky**2
    u_hat = np.fft.fft2(u)
    u_hat *= (ikx2+iky2) * (4.0 * np.pi**2)/(params.Lx*params.Ly)
    return np.real(np.fft.ifft2(u_hat))

def grad(u, params):
    ikx, iky = get_fft_coef(u)
    u_hat = np.fft.fft2(u) 
    ux = np.real(np.fft.ifft2(u_hat * ikx)) * (2.0 * np.pi)/params.Lx
    uy = np.real(np.fft.ifft2(u_hat * iky)) * (2.0 * np.pi)/params.Ly
    return ux, uy

def gradabs(u, params):
    ux, uy = grad(u, params)
    return np.sqrt(ux**2 + uy**2)

def div(ux, uy, params):
    ikx, iky = get_fft_coef(ux)
    ux_hat = np.fft.fft2(ux)
    uy_hat = np.fft.fft2(uy)
    u1 = np.real(np.fft.ifft2(ux_hat * ikx)) * (2.0 * np.pi)/params.Lx
    u2 = np.real(np.fft.ifft2(uy_hat * iky)) * (2.0 * np.pi)/params.Ly
    return (u1 + u2)

def diffusion_coef(x, y, freq, scale, torch_tensor=False):
    if not torch_tensor:
        return scale * (1 + np.sin(2*np.pi*freq*x) * np.sin(2*np.pi*freq*y))
    else:
        return scale * (1 + torch.sin(2*np.pi*freq*x) * torch.sin(2*np.pi*freq*y))

def diffusion_op(u, k, K, nx, ny, params):
    u = u.reshape((nx, ny))    
    gradux, graduy = grad(u, params)
    # diff tensor
    Kux = K['k11']*gradux + K['k12']*graduy
    Kuy = K['k22']*graduy + K['k12']*gradux
    # heterogeneous
    Kux *= k
    Kuy *= k
    return div(Kux, Kuy, params)

def rbf(x, y, p, sigma=1/64, spacing=2/64, ng=256, center=(0.5,0.5), torch_tensor=False):
    """ create an rbf grid basis functions """
    num = np.sqrt(ng) # num of centers in each direction
    l = (num - 1) * spacing # length of grid in each direction

    centers_x = np.arange(center[0]-l/2, center[0]+l/2+spacing, spacing)
    centers_y = np.arange(center[1]-l/2, center[1]+l/2+spacing, spacing)
    centers_x, centers_y = np.meshgrid(centers_x, centers_y)
    ratio = []
    c = []
    
    for cx, cy in zip(centers_x.flatten(), centers_y.flatten()):
        r = (x - cx)**2 + (y - cy)**2
        R = 2 * sigma**2
        ratio.append(r/R)
        c.append(1./(2*np.pi*sigma**2)) # normalizing factor

    source = 0*ratio[0]
    # compute for data; need dc component here
    idx = 0
    for r, ci in zip(ratio, c):
        source += p[idx]*ci*np.exp(-r)
        idx += 1
    dc = np.mean(source.flatten())
    source = source - dc

    return source


def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def compute_grad_norm(p_list):
    grad_norm = 0
    for p in p_list:
        param_g_norm = p.grad.detach().data.norm(2)
        grad_norm += param_g_norm.item()**2
    grad_norm = grad_norm**0.5
    return grad_norm

def l2_err(pred, target):
    x = torch.sum((pred-target)**2, dim=(-1,-2))/torch.sum(target**2, dim=(-1,-2)) 
    x = torch.sqrt(x)
    return torch.mean(x, dim=0)

def compute_err(output, target):
    err = output - target
    return np.linalg.norm(err[:])/np.linalg.norm(target[:])

def show(u, ax, fig, rescale=None):
    if u is not None:
        if rescale is None:
            h = ax.imshow(u.T, interpolation='nearest', cmap='rainbow',
                        origin='lower', aspect='auto')
        else:
            h = ax.imshow(u.T, interpolation='nearest', cmap='rainbow',
                        origin='lower', aspect='auto', vmin=rescale[0], vmax=rescale[1])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)
        cbar = fig.colorbar(h, cax=cax)
        cbar.ax.tick_params(labelsize=15)
        ax.tick_params(labelsize=15)


def vis_fields(fields, params, domain):
    source, target, pred, pde_res, temp = fields
    err = np.abs(pred - target)
    x_g = domain.x_g
    y_g = domain.y_g
    scale = [params.ny/params.Ly, params.nx/params.Lx]

    fx = 17 
    fy = 8
    fig = plt.figure(figsize=(fx,fy))
    ax1 = fig.add_subplot(2,3,1)
    show(source, ax1, fig)
    ax1.contour(y_g*scale[0], x_g*scale[1], source, 15)
    ax1.set_title("source")
    ax2 = fig.add_subplot(2,3,2)
    show(target, ax2, fig)
    ax2.set_title("target")
    ax2.contour(y_g*scale[0], x_g*scale[1], target, 15)
    ax3 = fig.add_subplot(2,3,3)
    show(pred, ax3, fig)
    ax3.set_title("pred")
    ax3.contour(y_g*scale[0], x_g*scale[1], pred, 15)
    ax4 = fig.add_subplot(2,3,4)
    show(pde_res, ax4, fig)
    ax4.set_title("pde-res")
#    ax4.contour(y_g*scale[0], x_g*scale[1], temp, 15)
    ax5 = fig.add_subplot(2,3,5)
    show(temp, ax5, fig)
    ax5.set_title("temp")
#    ax5.contour(y_g*scale[0], x_g*scale[1], temp, 15)
    ax6 = fig.add_subplot(2,3,6)
    show(err, ax6, fig)
    ax6.set_title("err")
    ax6.contour(y_g*scale[0], x_g*scale[1], err, 15)

    fig.tight_layout()

    return fig

def vis_fields_many(fields, params, domain):
    x_g = domain.x_g
    y_g = domain.y_g
    scale = [params.ny/params.Ly, params.nx/params.Lx]

    fx = 30 
    fy = 17
    fig = plt.figure(figsize=(fx,fy))
    num_examples = len(fields)//3
    n_cols = 4
    for i in range(num_examples):
        ax1 = fig.add_subplot(num_examples,n_cols,n_cols*i+1)
        source = fields[3*i]
        show(source, ax1, fig)
        ax1.contour(y_g*scale[0], x_g*scale[1], source, 15)
        ax1.set_title("source")
        ax2 = fig.add_subplot(num_examples,n_cols,n_cols*i+2)
        target = fields[3*i+1]
        show(target, ax2, fig)
        ax2.set_title("target")
        ax2.contour(y_g*scale[0], x_g*scale[1], target, 15)
        ax3 = fig.add_subplot(num_examples,n_cols,n_cols*i+3)
        pred = fields[3*i+2]
        show(pred, ax3, fig)
        ax3.contour(y_g*scale[0], x_g*scale[1], pred, 15)
        ax3.set_title("pred")
        ax4 = fig.add_subplot(num_examples,n_cols,n_cols*i+4)
        err = np.abs(pred-target)
        show(err, ax4, fig)
        ax4.contour(y_g*scale[0], x_g*scale[1], err, 15)
        ax4.set_title("err")

    fig.tight_layout()

    return fig

def set_activation(activation):
    if activation == 'identity':
        return nn.Identity()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'relu':
        return  nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    else:
        print("WARNING: invalid activation function!")
        return -1
