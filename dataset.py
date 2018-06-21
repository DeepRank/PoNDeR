import torch.utils.data as data

class PDB(data.Dataset):
    def __init__(self, root_dir, transform = None):
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return TODO

    def __getitem__(self, idx)
        TODO