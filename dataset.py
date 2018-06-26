import torch.utils.data as data
import pickle

class PDB(data.Dataset):
    def __init__(self, root_dir, train = True, transform = None, num_points=2500):
        self.root_dir = root_dir
        self.transform = transform

        pc_file = '/home/lukas/test.pickle'
        with open(pc_file, "rb") as f:
            self.irmsds, self.pcs = pickle.load(f)
    def __len__(self):
        return len(self.irmsds)

    def __getitem__(self, idx):
        return self.pcs[idx], self.pcs[idx]