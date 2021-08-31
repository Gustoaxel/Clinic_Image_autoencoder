from pathlib import Path
from typing import Tuple

import numpy as np
import torch as T
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import os


class ImageFolder720p(Dataset):
    """
    Image shape is (720, 1280, 3) --> (768, 1280, 3) --> 6x10 128x128 patches
    """

    def __init__(self, root: str):
        self.files = sorted(Path(root).iterdir())

    def __getitem__(self, index: int) -> Tuple[T.Tensor, np.ndarray, str]:
        path = str(self.files[index % len(self.files)])
        img = np.array(Image.open(path))

        pad = ((24, 24), (0, 0), (0, 0))

        # img = np.pad(img, pad, 'constant', constant_values=0) / 255
        img = np.pad(img, pad, mode="edge") / 255.0

        img = np.transpose(img, (2, 0, 1))
        img = T.from_numpy(img).float()

        patches = np.reshape(img, (3, 6, 128, 10, 128))
        patches = np.transpose(patches, (0, 1, 3, 2, 4))

        return img, patches, path

    def __len__(self):
        return len(self.files)
    
    
    
class ImagePlace(Dataset):
    """
    Image shape is (720, 1280, 3) --> (768, 1280, 3) --> 6x10 128x128 patches
    """

    def __init__(self, root: str):
        self.root_dir = root
        self.label = pd.read_csv(root+"/annotation.csv", delimiter=";")

    def __getitem__(self, idx: int) -> Tuple[T.Tensor, int, str]:
        
        if T.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.root_dir + self.label.iloc[idx, 0]

        

        img = np.array(Image.open(img_name))

        img = img / 255.0

        img = np.transpose(img, (2, 0, 1))
        img = T.from_numpy(img).float()
        
      

        return img, T.tensor(int(self.label.iloc[idx, 1]))

    def __len__(self):
        return len(self.label)
