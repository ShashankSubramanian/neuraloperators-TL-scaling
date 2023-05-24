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

def get_random_diffusion_tensor():
    ''' create a random diff tensor by controlling eigenvalues '''
    e1 = 1
    e2 = 5
    a1 = 1
    a4 = e1 + np.random.rand() * (e2 - e1) # random btw eigvals e1 and e2
    
    theta = np.random.rand() * 2 * np.pi # random rotation
    rot = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]) 
    theta_neg = -1 * theta
    rot_neg = np.array([[np.cos(theta_neg),-np.sin(theta_neg)],[np.sin(theta_neg),np.cos(theta_neg)]])
    A = np.array([[a1, 0], [0, a4]])
    A = rot_neg @ (A @ rot) # similarity transf to preserve eigs
    
    return A

def get_random_velocity_vector():
    ''' create a random velocity vector by sampling on a circle '''
    theta = np.random.rand() * 2 * np.pi
    r = 1
    return [r*np.cos(theta), r*np.sin(theta)]

def advection_op(u, v, nx, ny, params):
    ''' does v.grad(u) '''
    u = u.reshape((nx, ny))    
    gradux, graduy = grad(u, params)
    return (v['v1']*gradux + v['v2']*graduy)

def diffusion_op(u, k, K, nx, ny, params):
    ''' does div(Kgradu) '''
    u = u.reshape((nx, ny))    
    gradux, graduy = grad(u, params)
    # diff tensor
    Kux = K['k11']*gradux + K['k12']*graduy
    Kuy = K['k22']*graduy + K['k12']*gradux
    # heterogeneous
    Kux *= k
    Kuy *= k
    return div(Kux, Kuy, params)

def advdiff(x, y, std, space, vf, params):
    """ -v.\gradu + \lapu = -f """
    k_mat = get_random_diffusion_tensor()       
    K = {'k11': k_mat[0,0], 'k22': k_mat[1,1], 'k12': k_mat[0,1]}
    vel = get_random_velocity_vector()
    v = {'v1': vel[0], 'v2': vel[1]} # just for readability make a dict

    ad1 = params.ad1
    ad2 = params.ad2
    adr = ad1 + np.random.rand() * (ad2 - ad1)
    # sample lamda based on adr
    lams = np.load("utils/lambda.npy")
    ads = np.load("utils/ads.npy")
    means = np.mean(ads, axis=1)
    lam = lams[closest(means, adr)]

    params.lam = lam

    x_g, y_g = np.meshgrid(x, y)

    # create a source function
    ng = params.ng
    p = np.zeros(ng)
        
    min_act = 1E-3 # avoid zeros
    max_act = 1
        
    for i in range(ng):
        alpha = np.random.rand()
        if alpha > vf:
            p[i] = (max_act - min_act) * np.random.rand() + min_act

    all_zeros = not np.any(p)
    if all_zeros:
        randidx = np.random.randint(0, len(p))
        p[randidx] = (max_act - min_act) * np.random.rand() + min_act

    
    source = rbf(x_g, y_g, p, ng=ng, sigma=std, spacing=space) # linear combination of rbfs

    nx = x.shape[0]
    ny = y.shape[0]
    ikx = fft_coef(nx).reshape(1,nx)
    ikx = np.repeat(ikx, ny, axis=0)
    iky = fft_coef(ny).reshape(ny,1)
    iky = np.repeat(iky, nx, axis=1)
    ikx2 = ikx**2
    iky2 = iky**2
    

    f_hat = np.fft.fft2(source) # RHS
    
    diff_factor = ikx2*K['k11'] + iky2*K['k22'] + 2*ikx*iky*K['k12']
    diff_factor *= (4.0 * np.pi**2) / (params.Lx * params.Ly) # not a 2pi domain, but unit cube
    adv_factor = v['v1']*ikx + v['v2']*iky
    adv_factor *= (2.0 * np.pi) / (params.Lx)  # implicity assumed Lx = Ly; TODO: be careful
    factor = (1 - params.lam) * params.diff_coef_scale * diff_factor - params.lam * params.adv_coef_scale * adv_factor

    factor = np.where(factor == 0, 0, -1/factor) # zeroth mode; set to zero 
    u_hat = factor * f_hat
    
    u = np.real(np.fft.ifft2(u_hat))
    u = u - np.mean(u.flatten())  # remove the dc component

    au = (params.lam) * params.adv_coef_scale * advection_op(u, v, nx, ny, params)
    du = (1 - params.lam) * params.diff_coef_scale * diffusion_op(u, 1, K, nx, ny, params)
    au = np.linalg.norm(au)
    du = np.linalg.norm(du)
    ratio = au/du
    
     
    k = np.array([K['k11'], K['k22'], K['k12']])
    v = np.array([v['v1'], v['v2']]) # for dataset creating and interfacing with training code

    return u, source, k, v, ratio


def create_hdf5(path, dat, ten, other):
    with h5py.File(path, "a") as f:
        try:
            f.create_dataset('fields', dat.shape, dtype='<f4', data=dat)
            f.create_dataset('tensor', ten.shape, dtype='<f4', data=ten)
            f.create_dataset('other', other.shape, dtype='<f4', data=other)
        except:
            del f['fields']
            del f['tensor']
            del f['other']
            f.create_dataset('fields', dat.shape, dtype='<f4', data=dat)
            f.create_dataset('tensor', ten.shape, dtype='<f4', data=ten)
            f.create_dataset('other', other.shape, dtype='<f4', data=other)

def num_ex(ntrain, idx, num_vfs):
    nex = (ntrain//num_vfs if idx is not num_vfs-1 else ntrain - (num_vfs-1)*(ntrain//num_vfs))
    return nex

def closest(lst, k):
    lst = np.asarray(lst)
    idx = (np.abs(lst - k)).argmin()
    return idx

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", default=100, type=int, help="number of training examples")
    parser.add_argument("--nval", default=25, type=int, help="number of validation examples")
    parser.add_argument("--ntest", default=25, type=int, help="number of testing examples")
    parser.add_argument("--n", default=128, type=int, help="grid size nxn")
    parser.add_argument("--ng", default=144, type=int, help="number of gaussians in grid")
    parser.add_argument("--sparse", action='store_true', help="creates sparse gaussians")
    parser.add_argument("--datapath", default="./", type=str, help="path to root dir to store data")
    parser.add_argument("--adr1", default=0.2, type=float, help="sample AD ratios starting from adr1")
    parser.add_argument("--adr2", default=1, type=float, help="sample AD ratios  ending at adr2")

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
    params["Lx"] = lx
    params["Ly"] = ly
    params["adv_coef_scale"] = 1
    params["diff_coef_scale"] = 0.01
    params['ad1'] = args.adr1
    params['ad2'] = args.adr2
    params["ng"] = args.ng
    params['sparse'] = args.sparse
    params = SimpleNamespace(**params)
    std = 1/32
    space = 2*std
    x_g, y_g = np.meshgrid(x, y)

    train = np.zeros((ntrain, 2, x.shape[0], y.shape[0]))
    train_ten = np.zeros((ntrain, 5))
    train_ratios = np.zeros((ntrain, 1))
    val = np.zeros((nval, 2, x.shape[0], y.shape[0]))
    val_ten = np.zeros((nval, 5))
    val_ratios = np.zeros((nval, 1))
    test = np.zeros((ntest, 2, x.shape[0], y.shape[0]))
    test_ten = np.zeros((ntest, 5))
    test_ratios = np.zeros((ntest, 1))

    vfs = [0.2, 0.4, 0.6, 0.8]
    num_vfs = len(vfs)
    print_freq = 100
    t0 = time.time()
    sim_train = sim_val = sim_test = 0
    for idx, vf in enumerate(vfs):
        nex = num_ex(ntrain, idx, num_vfs)
        print("For vf {}, there are {} train examples".format(vf, nex))
        for s in range(nex):
            u, source, k, v, ratios = advdiff(x, y, std, space, vf, params)
            train[sim_train,0,...] = source
            train[sim_train,1,...] = u
            train_ten[sim_train, 0:3] = k * params.diff_coef_scale * (1 - params.lam)
            train_ten[sim_train, 3:5] = v * params.adv_coef_scale * params.lam
            train_ratios[sim_train] = ratios
            sim_train += 1
            if s%print_freq==0:
                print("finished {} train for vf {}".format(s, vf))
        nex = num_ex(nval, idx, num_vfs)
        print("For vf {}, there are {} val examples".format(vf, nex))
        for s in range(nex):
            u, source, k, v, ratios = advdiff(x, y, std, space, vf, params)
            val[sim_val,0,...] = source
            val[sim_val,1,...] = u
            val_ten[sim_val, 0:3] = k * params.diff_coef_scale * (1 - params.lam)
            val_ten[sim_val, 3:5] = v * params.adv_coef_scale * params.lam
            val_ratios[sim_val] = ratios
            sim_val += 1
        nex = num_ex(ntest, idx, num_vfs)
        print("For vf {}, there are {} test examples".format(vf, nex))
        for s in range(nex):
            u, source, k, v, ratios = advdiff(x, y, std, space, vf, params)
            test[sim_test,0,...] = source
            test[sim_test,1,...] = u
            test_ten[sim_test, 0:3] = k * params.diff_coef_scale * (1 - params.lam)
            test_ten[sim_test, 3:5] = v * params.adv_coef_scale * params.lam
            test_ratios[sim_test] = ratios
            sim_test += 1
            if s%print_freq==0:
                print("finished {} test".format(sim_test))

    print("time = {}".format(time.time() - t0))
    datapath = args.datapath
    print("saving files to {}".format(datapath))

    create_hdf5(os.path.join(datapath, "_train_adr{}_{}_32k.h5".format(params.ad1,params.ad2)), train, train_ten, train_ratios)
    create_hdf5(os.path.join(datapath, "_val_adr{}_{}_4k.h5".format(params.ad1,params.ad2)), val, val_ten, val_ratios)
    create_hdf5(os.path.join(datapath, "_test_adr{}_{}_4k.h5".format(params.ad1,params.ad2)), test, test_ten, test_ratios)
