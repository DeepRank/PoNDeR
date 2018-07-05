import torch
import torch.utils.data as data
import os
import random
import numpy as np
import h5py

# No more than one worker can be used for these types of dataset as HDF5 does not multithread appropriately

class PDBset(data.Dataset):
    def __init__(self, hdf5_file, num_points, group='train', metric='dockQ'):
        self.hf = h5py.File(hdf5_file,'r')
        self.num_points = num_points
        self.group = self.hf[group]
        self.keys = list(self.group.keys())
        self.metric = metric

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        ds = self.group.get(self.keys[idx])
        pc = np.array(ds)
        mtrc = ds.attrs[self.metric]

        pc = samplePoints(pc, self.num_points)

        return torch.from_numpy(pc), np.float32(mtrc)

class DualPDBset(data.Dataset):
    def __init__(self, hdf5_file, num_points, group='train', metric='dockQ'):
        self.hf = h5py.File(hdf5_file,'r')
        self.num_points = num_points
        self.group = self.hf[group]
        self.keys = list(self.group.keys())
        self.metric = metric

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        subgroup = self.group.get(self.keys[idx])
        pcA = np.array(subgroup.get('A'))
        pcB = np.array(subgroup.get('B'))
        mtrc = subgroup.attrs[self.metric]

        pcA = samplePoints(pcA, self.num_points)
        pcB = samplePoints(pcB, self.num_points)

        pc = np.concatenate((pcA, pcB), axis=0) # Concatenate to conform with pytorch API (nn.module takes only one input)

        return torch.from_numpy(pc), np.float32(mtrc)


# Zero concatenation, safe for both maxpooling and avgpooling
def samplePoints(pc, num_points):
    if len(pc) < num_points:
        zeros = np.zeros((num_points-len(pc), pc.shape[1]),dtype=np.float32)
        pc = np.concatenate((pc, zeros), axis=0)
    else:
        point_ids = random.sample(range(len(pc)), num_points)
        pc = np.take(pc, point_ids, axis=0)
    return pc