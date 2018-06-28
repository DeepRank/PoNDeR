import torch
import torch.utils.data as data
import pickle
import os
import random
import numpy as np

class PDBset(data.Dataset):
    def __init__(self, root_dir, train = True, num_points=250):
        self.root_dir = root_dir
        self.num_points = num_points
        if train:
            self.subfolder = 'train/'
        else:
            self.subfolder = 'test/'

        self.file_list = os.listdir(root_dir + subfolder)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        with open(self.root_dir + self.subfolder + self.file_list[idx], "rb") as f:
            irmsd, pc = pickle.load(f)

            if len(pc) < self.num_points:
                point_ids = random.sample(range(len(pc)), len(pc)) + random.sample(range(len(pc)), self.num_points-len(pc))
            else:
                point_ids = random.sample(range(len(pc)), self.num_points)

            pc = np.take(pc,point_ids,axis=0)
            
        return torch.from_numpy(pc), irmsd