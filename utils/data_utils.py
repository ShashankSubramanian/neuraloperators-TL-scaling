"""
  data loaders
"""
import re
import time
import os, sys
import logging
import h5py
import glob
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.distributed import DistributedSampler


def get_data_loader(params, location, distributed, train=True, pack=False):
    transform = torch.from_numpy
    dataset = PDESolns(params, location, transform, train)
    sampler = DistributedSampler(dataset, shuffle=train) if distributed else None
    if train:
        batch_size = params.local_batch_size
    else:
        batch_size = params.local_valid_batch_size
    if not pack:
        dataloader = DataLoader(dataset,
                                batch_size=int(batch_size),
                                num_workers=params.num_data_workers,
                                shuffle=False,#(sampler is None),
                                sampler=sampler,
                                drop_last=True,
                                pin_memory=torch.cuda.is_available())
    else:
        # data is small, pack it all onto the gpu
        X = dataset.data[:,0:dataset.in_channels]
        y = dataset.data[:,dataset.in_channels:]
        X = torch.tensor(X, requires_grad=True).float().to(params.device)
        y = torch.tensor(y, requires_grad=True).float().to(params.device)
        tensor_dataset = TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(tensor_dataset, batch_size=int(batch_size), shuffle=True)
    return dataloader, dataset, sampler


class PDESolns(Dataset):
    def __init__(self, params, location, transform, train):
        self.transform = transform
        self.params = params
        self.location = location
        self.train = train
        if hasattr(self.params, "subsample") and (self.train):
            self.subsample = self.params.subsample
        else:
            self.subsample = 1 # subsample only if training
        self.scales = None
        self._get_files_stats()
        file = self._open_file(self.location)
        self.data = file['fields']
        if 'tensor' in list(file.keys()):
            self.tensor = file['tensor']
        else:
            self.tensor = None

    def _get_files_stats(self):
        self.file = self.location
        with h5py.File(self.file, 'r') as _f:
            logging.info("Getting file stats from {}".format(self.file))
            self.n_samples = _f['fields'].shape[0]
            self.img_shape_x = _f['fields'].shape[2]
            self.img_shape_y = _f['fields'].shape[3]
            self.in_channels = _f['fields'].shape[1]-1
            if 'tensor' in list(_f.keys()):
                self.tensor_shape = _f['tensor'].shape[1]
            else:
                self.tensor_shape = 0
        self.n_samples /= self.subsample
        self.n_samples = int(self.n_samples)
        logging.info("Found data at path {}. Number of examples: {}. Image Shape: {} x {}".format(self.location, self.n_samples, self.img_shape_x, self.img_shape_y))
        if hasattr(self.params, "scales_path"):
            self.scales = np.load(self.params.scales_path)
            self.scales = np.array([s if s != 0 else 1 for s in self.scales]) 
            self.scales = self.scales.astype('float32')
            measure_x = self.scales[-2] / self.img_shape_x
            measure_y = self.scales[-1] / self.img_shape_y
            self.measure = measure_x * measure_y
            logging.info("Scales for PDE are (source, tensor, sol, domain): {}".format(self.scales))
            logging.info("Measure of the set is lx/nx * ly/ny =  {}/{} * {}/{}".format(self.scales[-2], self.img_shape_x, self.scales[-1], self.img_shape_y))

    def __len__(self):
        return self.n_samples

    def _open_file(self, path):
        return h5py.File(path, 'r')

    def __getitem__(self, idx):
        local_idx = int(idx*self.subsample)
        X = (self.data[local_idx,0:self.in_channels])
        if self.tensor: # append coefficient tensor to channels
            tensor = []
            for tidx in range(self.tensor_shape):
                coef = np.full((1, self.img_shape_x, self.img_shape_y), self.tensor[local_idx,tidx])
                tensor.append(coef)
            X = np.concatenate([X] + tensor, axis=0).astype('float32')

        if self.scales is not None:
            f_norm = np.linalg.norm(X[0]) * self.measure
            f_scaling = f_norm / self.scales[0]
            X = X / f_scaling # ensures that 10f and 10k for example, have the same input
            # scale the tensors
            X[self.in_channels:] = X[self.in_channels:] / self.scales[self.in_channels:(self.in_channels + self.tensor_shape), None, None]

        X = self.transform(X)
        y = self.transform(self.data[local_idx,self.in_channels:])
        return X, y


