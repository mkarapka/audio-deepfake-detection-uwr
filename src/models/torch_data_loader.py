import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.common.utils import get_device


class TorchDataLoader(Dataset):
    def __init__(self, metadata: pd.DataFrame, features: np.ndarray, transform=None, device: str = None):
        self.device = get_device() if device is None else device
        self.metadata = metadata.reset_index(drop=True)
        self.features = features
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int):
        x = self.features[idx]
        y = int(self.metadata["label"].iloc[idx] == "bonafide")

        if self.transform is not None:
            x = self.transform(x)

        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        return x_tensor, y_tensor
