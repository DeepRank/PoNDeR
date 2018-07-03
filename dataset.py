import torch
import torch.utils.data as data
import pickle
import os
import random
import numpy as np
import h5py

# No more than one worker can be used for this type of dataset as HDF5 does not multithread appropriately

class PDBset(data.Dataset):
    def __init__(self, hdf5_file, num_points, group='train'):
        self.hf = h5py.File(hdf5_file,'r')
        self.num_points = num_points
        self.group = self.hf[group]
        self.keys = list(self.group.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        ds = self.group.get(self.keys[idx])
        pc = np.array(ds)
        dockQ = ds.attrs['dockQ']

        # Duplicate points within pointcloud don't matter in PointNet architecture due to maxpooling
        # This does NOT apply to avgpooling
        if len(pc) < self.num_points:
            point_ids = random.sample(range(len(pc)), len(pc))
            while len(point_ids)<self.num_points-len(pc):
                point_ids += random.sample(range(len(pc)), len(pc))
            point_ids += random.sample(range(len(pc)), self.num_points-len(point_ids))
        else:
            point_ids = random.sample(range(len(pc)), self.num_points)

        pc = np.take(pc, point_ids, axis=0)

        return torch.from_numpy(pc), np.float32(dockQ)
