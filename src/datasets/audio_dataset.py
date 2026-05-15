import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self, metadata: pd.DataFrame, features: np.ndarray, transform=None, device: str = "cpu"):
        self.device = device
        self.metadata = metadata.reset_index(drop=True)
        self.features = features
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int):
        x = self.features[idx]
        y = int(self.metadata["target"].iloc[idx] == "bonafide")

        if self.transform is not None:
            x = self.transform(x)

        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor([y], dtype=torch.float32)

        if self.device != "cpu":
            x_tensor = x_tensor.to(self.device)
            y_tensor = y_tensor.to(self.device)

        return x_tensor, y_tensor
