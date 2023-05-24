import os, sys, time
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
from utils.misc_utils import show, fft_coef, grad, div
from types import SimpleNamespace
import random
import h5py
import time

def rbf(x, y, p, sigma=1/64, spacing=2/64, ng=256, center=(0.5,0.5)):
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
    
    idx = 0
    for r, ci in zip(ratio, c):
        source += p[idx]*1*np.exp(-r)
        idx += 1

    source = source/np.max(source)
        
    dc = np.mean(source.flatten())
    source = source - dc # remove integral of source

    return source

def get_random_diffusion_tensor(e1, e2):
    a1 = 1
    a4 = e1 + np.random.rand() * (e2 - e1)
    
    theta = np.random.rand() * 2 * np.pi
    rot = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]) 
    theta_neg = -1 * theta
    rot_neg = np.array([[np.cos(theta_neg),-np.sin(theta_neg)],[np.sin(theta_neg),np.cos(theta_neg)]])
    A = np.array([[a1, 0], [0, a4]])
    A = rot_neg @ (A @ rot) # similarity transf to preserve eigs

    return A

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
    
def poisson(x, y, std, space, vf, params):
    k_mat = get_random_diffusion_tensor(params.e1, params.e2)
    K = {'k11': k_mat[0,0], 'k22': k_mat[1,1], 'k12': k_mat[0,1]}

    x_g, y_g = np.meshgrid(x, y)
    ng = params.ng
    p = np.zeros(ng)
    min_act = 1E-3 # avoid zeros
    max_act = 1
    if not params.sparse:
        p = np.random.rand(ng,1)
    for i in range(ng):
        alpha = np.random.rand()
        if alpha > vf:
            p[i] = (max_act - min_act) * np.random.rand() + min_act

    all_zeros = not np.any(p)
    if all_zeros:
        randidx = np.random.randint(0, len(p))
        p[randidx] = (max_act - min_act) * np.random.rand() + min_act

    source = rbf(x_g, y_g, p, ng=ng, sigma=std, spacing=space)

    nx = x.shape[0]
    ny = y.shape[0]
    ikx = fft_coef(nx).reshape(1,nx)
    ikx = np.repeat(ikx, ny, axis=0)
    iky = fft_coef(ny).reshape(ny,1)
    iky = np.repeat(iky, nx, axis=1)
    ikx2 = ikx**2
    iky2 = iky**2

    f_hat = np.fft.fft2(source)

    diff_factor = ikx2*K['k11'] + iky2*K['k22'] + 2*ikx*iky*K['k12']
    diff_factor *= (4.0 * np.pi**2) / (params.Lx * params.Ly) # not a 2pi domain, but unit cube
    factor = params.diff_coef_scale * diff_factor
    factor = np.where(factor == 0, 0, -1/factor) # zeroth mode; set to zero
    u_hat = factor * f_hat

    u = np.real(np.fft.ifft2(u_hat))
    u = u - np.mean(u.flatten())  # remove the dc component

    k = np.array([K['k11'], K['k22'], K['k12']])

    return u, source, k


def create_hdf5(path, dat, ten):
    with h5py.File(path, "a") as f:
        try:
            f.create_dataset('fields', dat.shape, dtype='<f4', data=dat)
            f.create_dataset('tensor', ten.shape, dtype='<f4', data=ten)
        except:
            del f['fields']
            del f['tensor']
            f.create_dataset('fields', dat.shape, dtype='<f4', data=dat)
            f.create_dataset('tensor', ten.shape, dtype='<f4', data=ten)

def num_ex(ntrain, idx, num_vfs):
    nex = (ntrain//num_vfs if idx is not num_vfs-1 else ntrain - (num_vfs-1)*(ntrain//num_vfs))
    return nex

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", default=100, type=int, help="number of training examples")
    parser.add_argument("--nval", default=25, type=int, help="number of validation examples")
    parser.add_argument("--ntest", default=25, type=int, help="number of testing examples")
    parser.add_argument("--n", default=128, type=int, help="grid size nxn")
    parser.add_argument("--ng", default=144, type=int, help="number of gaussians in grid")
    parser.add_argument("--sparse", action='store_true', help="creates sparse gaussians")
    parser.add_argument("--datapath", default="./", type=str, help="path to root dir to store data")
    parser.add_argument("--e1", default=1, type=int, help="sample diffusion eigenvalues starting from e1")
    parser.add_argument("--e2", default=5, type=int, help="sample diffusion eigenvalues ending at e2")

    args = parser.parse_args()
    print(args)
    seed = 0

    random.seed(seed)
    np.random.seed(seed)

    ntrain = args.ntrain
    nval = args.nval
    ntest = args.ntest
    nx = ny = args.n
    lx = ly = 1
    dx = lx/nx
    dy = ly/ny
    x = np.arange(0, lx, dx)
    y = np.arange(0, ly, dy)
    params = {}
    params["diff_coef_freq"] = 0
    params["dc"] = 0
    params["Lx"] = lx
    params["Ly"] = ly

    params["diff_coef_scale"] = 0.01
    params["e1"] = args.e1
    params["e2"] = args.e2
    
    params["ng"] = args.ng
    params['sparse'] = args.sparse
    params = SimpleNamespace(**params)

    print("Sampling diffusion from {} to {}".format(params.e1, params.e2))
    std = 1/32
    space = 2*std
    x_g, y_g = np.meshgrid(x, y)

    train = np.zeros((ntrain, 2, x.shape[0], y.shape[0]))
    train_k = np.zeros((ntrain, 3))
    val = np.zeros((nval, 2, x.shape[0], y.shape[0]))
    val_k = np.zeros((nval, 3))
    test = np.zeros((ntest, 2, x.shape[0], y.shape[0]))
    test_k = np.zeros((ntest, 3))

    vfs = [0.2, 0.4, 0.6, 0.8]
    num_vfs = len(vfs)
    print_freq = 100
    t0 = time.time()
    sim_train = sim_val = sim_test = 0
    for idx, vf in enumerate(vfs):
        nex = num_ex(ntrain, idx, num_vfs)
        print("For vf {}, there are {} train examples".format(vf, nex))
        for s in range(nex):
            u, source, k = poisson(x, y, std, space, vf, params)
            train[sim_train,0,...] = source
            train[sim_train,1,...] = u
            train_k[sim_train] = k * params.diff_coef_scale
            sim_train += 1
            if s%print_freq==0:
                print("finished {} train for vf {}".format(s, vf))
        nex = num_ex(nval, idx, num_vfs)
        print("For vf {}, there are {} val examples".format(vf, nex))
        for s in range(nex):
            u, source, k = poisson(x, y, std, space, vf, params)
            val[sim_val,0,...] = source
            val[sim_val,1,...] = u
            val_k[sim_val] = k * params.diff_coef_scale
            sim_val += 1
        nex = num_ex(ntest, idx, num_vfs)
        print("For vf {}, there are {} test examples".format(vf, nex))
        for s in range(nex):
            u, source, k = poisson(x, y, std, space, vf, params)
            test[sim_test,0,...] = source
            test[sim_test,1,...] = u
            test_k[sim_test] = k * params.diff_coef_scale
            sim_test += 1
            if s%print_freq==0:
                print("finished {} test".format(sim_test))

    print("time = {}".format(time.time() - t0))
    datapath = args.datapath 

    print("saving files to {}".format(datapath))

    create_hdf5(os.path.join(datapath, "_train_k{}_{}_32k.h5".format(params.e1,params.e2)), train, train_k)
    create_hdf5(os.path.join(datapath, "_val_k{}_{}_4k.h5".format(params.e1,params.e2)), val, val_k)
    create_hdf5(os.path.join(datapath, "_test_k{}_{}_4k.h5".format(params.e1,params.e2)), test, test_k)
