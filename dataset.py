import torch
import torch.utils.data as data
import os
import random
import numpy as np
import h5py
import math

# No more than one worker can be used for these types of dataset as HDF5 does not multithread appropriately

LOGCUTOFF = -3.73 # Cutoff for log

class PDBset(data.Dataset):
    def __init__(self, hdf5_file, num_points, group='train', metric='dockQ', log=False, classification=False):
        self.hf = h5py.File(hdf5_file,'r')
        self.num_points = num_points
        self.group = self.hf[group]
        self.keys = list(self.group.keys())
        self.metric = metric
        self.log = log
        self.classification = classification

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        pc = self.group.get(self.keys[idx])
        mtrc = np.float32(pc.attrs[self.metric])
        pc = samplePoints(np.array(pc), self.num_points)
        if self.log:
            mtrc = math.log(mtrc)
            cutoff = LOGCUTOFF
        if self.classification:
            if mtrc < cutoff:
                mtrc = 0
            else:
                mtrc = 1
        return torch.from_numpy(pc), mtrc
    
    def getFeatWidth(self):
        return self.hf.attrs['feat_width'].item()

class DualPDBset(data.Dataset):
    def __init__(self, hdf5_file, num_points, group='train', metric='dockQ', log=False, classification=False):
        self.hf = h5py.File(hdf5_file,'r')
        self.num_points = num_points
        self.group = self.hf[group]
        self.keys = list(self.group.keys())
        self.metric = metric
        self.log = log
        self.classification = classification


    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        subgroup = self.group.get(self.keys[idx])
        pcA = np.array(subgroup.get('A'))
        pcB = np.array(subgroup.get('B'))
        mtrc = np.float32(subgroup.attrs[self.metric])

        pcA = samplePoints(pcA, self.num_points)
        pcB = samplePoints(pcB, self.num_points)

        pc = np.concatenate((pcA, pcB), axis=0) # Concatenate to conform with pytorch API (nn.module takes only one input)
        if self.log:
            mtrc = math.log(mtrc)
            cutoff = LOGCUTOFF
        if self.classification:
            if mtrc < cutoff:
                mtrc = 0
            else:
                mtrc = 1

        return torch.from_numpy(pc), mtrc
    
    def getFeatWidth(self):
        return self.hf.attrs['feat_width'].item()

# Zero concatenation, safe for maxpooling but not for avgpooling
def samplePoints(pc, num_points):
    if len(pc) < num_points:
        zeros = np.zeros((num_points-len(pc), pc.shape[1]),dtype=np.float32)
        pc = np.concatenate((pc, zeros), axis=0)
    else:
        point_ids = random.sample(range(len(pc)), num_points)
        pc = np.take(pc, point_ids, axis=0)
    return pc