import torch
import torch.utils.data as data
import pickle
import os
import random
import numpy as np

class PDBset(data.Dataset):
    def __init__(self, root_dir, train = True, num_points=250):
        self.file_list = os.listdir(root_dir)
        self.root_dir = root_dir
        self.num_points = num_points

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        pcs = []
        irmsds = []

        for i, val in enumerate(idx):
            with open(self.root_dir+self.file_list[val], "rb") as f:
                irmsd, pc = pickle.load(f)
                
                if len(pc) < self.num_points:
                    point_ids = random.sample(range(self.num_points), len(pc)) + random.sample(range(self.num_points), self.num_points-len(pc))
                else:
                    point_ids = random.sample(range(self.num_points), self.num_points)

                irmsds.append(irmsd)
                pcs.append(np.take(pc,point_ids,axis=0))
                
        return torch.from_numpy(np.stack(pcs)), irmsds