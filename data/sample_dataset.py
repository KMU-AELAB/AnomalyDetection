import os
from PIL import Image
import random
import numpy as np

import torch
from torch.utils.data import Dataset


class SampleDataset(Dataset):
    def __init__(self,config, torchvision_transform):
        self.root_dir = config.root_path
        self.data_list = os.listdir(os.path.join(self.root_dir, config.data_path))
        self.config = config
        self.transform = torchvision_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_name = os.path.join(self.root_dir, self.config.data_path, self.data_list[idx])
        data = Image.open(data_name)

        data = self.transform(data)

        return {'X': data}
