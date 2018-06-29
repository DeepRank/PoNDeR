import torch
import torch.utils.data as data
import pickle
import os
import random
import numpy as np


class PDBset(data.Dataset):
    def __init__(self, root_dir, num_points, train=True):
        self.root_dir = root_dir
        self.num_points = num_points
        if train:
            self.subfolder = 'train/'
        else:
            self.subfolder = 'test/'

        self.file_list = os.listdir(root_dir + self.subfolder)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        with open(self.root_dir + self.subfolder + self.file_list[idx], "rb") as f:
            irmsd, pc = pickle.load(f)
            # Duplicate points within pointcloud don't matter in PointNet architecture due to maxpooling
            if len(pc) < num_points:
                point_ids = random.sample(range(len(pc)), len(pc))
                while len(point_ids)<num_points-len(pc):
                    point_ids += random.sample(range(len(pc)), len(pc))
                point_ids += random.sample(range(len(pc)), num_points-len(point_ids))
            else:
                point_ids = random.sample(range(len(pc)), num_points)

            pc = np.take(pc, point_ids, axis=0)

        return torch.from_numpy(pc).float(), np.float32(irmsd)
