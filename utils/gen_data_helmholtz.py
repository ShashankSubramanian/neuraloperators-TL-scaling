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
from utils.misc_utils import show, fft_coef, grad, div, laplacian
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
        
    return source

def helm_op(u, omega,  nx, ny, params):
    u = u.reshape((nx, ny))
    lap = params.diff_coef_scale * laplacian(u, params)
    lap += omega * u
    return lap


def helmholtz(x, y, std, space, vf, params):
    x_g, y_g = np.meshgrid(x, y)

    sol_max = 100
    while (sol_max > 2): # ignore solutions with large values
        ng = params.ng
        p = np.zeros(ng)
        min_act = 1E-3 # avoid zeros
        max_act = 1

        omega = np.random.randint(params.o1, params.o2+1)

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

        ik_factor =  ikx2 + iky2
        ik_factor *= (4.0 * np.pi**2) / (params.Lx * params.Ly) * params.diff_coef_scale
        factor = (omega + ik_factor)
        condn = (factor == 0)

        f_hat = np.where(condn, 0, f_hat)
        source = source if np.all(~condn[:]) else np.real(np.fft.ifft2(f_hat))

        factor = np.where(condn, 0, -1/factor) # set to zero in undefined places in freq space
        u_hat = factor * f_hat

        u = np.real(np.fft.ifft2(u_hat))

        # check the result by seeing norm(LHS) == norm(RHS)
        lhs = helm_op(u, omega, nx, ny, params)
        lhs_norm = np.linalg.norm(lhs)
        rhs_norm = np.linalg.norm(source)
        if (np.abs(lhs_norm - rhs_norm) > 1E-5):
            print("INACCURATE SOLUTION!")

        sol_max = np.max(np.abs(u[:]))


    return u, source, omega


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
    parser.add_argument("--o1", default=1, type=int, help="sample wavenumbers starting from o1")
    parser.add_argument("--o2", default=10, type=int, help="sample wavenumbers  ending at o2")

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
    params["o1"] = args.o1
    params["o2"] = args.o2

    params["ng"] = args.ng
    params['sparse'] = args.sparse
    params = SimpleNamespace(**params)
    std = 1/32
    space = 2*std
    x_g, y_g = np.meshgrid(x, y)

    train = np.zeros((ntrain, 2, x.shape[0], y.shape[0]))
    train_om = np.zeros((ntrain, 2))
    val = np.zeros((nval, 2, x.shape[0], y.shape[0]))
    val_om = np.zeros((nval, 2))
    test = np.zeros((ntest, 2, x.shape[0], y.shape[0]))
    test_om = np.zeros((ntest, 2))

    vfs = [0.2, 0.4, 0.6, 0.8]
    num_vfs = len(vfs)
    in_distr = True
    print_freq = 100
    t0 = time.time()
    sim_train = sim_val = sim_test = 0
    for idx, vf in enumerate(vfs):
        nex = num_ex(ntrain, idx, num_vfs)
        print("For vf {}, there are {} train examples".format(vf, nex))
        for s in range(nex):
            u, source, om = helmholtz(x, y, std, space, vf, params)
            train[sim_train,0,...] = source
            train[sim_train,1,...] = u
            train_om[sim_train,0] = params.diff_coef_scale
            train_om[sim_train,1] = om
            sim_train += 1
            if s%print_freq==0:
                print("finished {} train for vf {}".format(s, vf))
        nex = num_ex(nval, idx, num_vfs)
        print("For vf {}, there are {} val examples".format(vf, nex))
        for s in range(nex):
            u, source, om = helmholtz(x, y, std, space, vf, params)
            val[sim_val,0,...] = source
            val[sim_val,1,...] = u
            val_om[sim_val,0] = params.diff_coef_scale
            val_om[sim_val,1] = om
            sim_val += 1
        nex = num_ex(ntest, idx, num_vfs)
        print("For vf {}, there are {} test examples".format(vf, nex))
        for s in range(nex):
            u, source, om = helmholtz(x, y, std, space, vf, params)
            test[sim_test,0,...] = source
            test[sim_test,1,...] = u
            test_om[sim_test,0] = params.diff_coef_scale
            test_om[sim_test,1] = om
            sim_test += 1
            if s%print_freq==0:
                print("finished {} test".format(sim_test))

    print("time = {}".format(time.time() - t0))
    datapath = args.datapath 

    print("saving files to {}".format(datapath))

    create_hdf5(os.path.join(datapath, "_train_o{}_{}_32k.h5".format(params.o1,params.o2)), train, train_om)
    create_hdf5(os.path.join(datapath, "_val_o{}_{}_4k.h5".format(params.o1,params.o2)), val, val_om)
    create_hdf5(os.path.join(datapath, "_test_o{}_{}_4k.h5".format(params.o1,params.o2)), test, test_om)
