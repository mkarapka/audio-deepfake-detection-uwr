import torch
from torch.utils.data import Dataset

from src.common.basic_functions import get_device
from src.common.constants import Constants as consts
from src.preprocessing.io.feature_loader import FeatureLoader


class DataLoader(Dataset):
    def __init__(
        self,
        split: str,
        feat_suffix: str,
        frac: float = 1.0,
        audio_id_sample: bool = False,
        file_name: str = consts.feature_extracted,
        device: str = None,
    ):
        self.device = get_device() if device is None else device
        feature_loader = FeatureLoader(file_name=file_name, feat_suffix=feat_suffix)
        self.metadata, self.features = feature_loader.load_data_split(split_name=split)
        if frac < 1.0:
            if audio_id_sample:
                self.metadata, self.features = feature_loader.sample_by_audio_ids(
                    metadata=self.metadata, features=self.features, fraction=frac
                )
            else:
                self.metadata, self.features = feature_loader.sample_data(
                    metadata=self.metadata, features=self.features, fraction=frac
                )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int):
        x = self.features[idx]
        y = int(self.metadata["label"].iloc[idx] == "bonafide")

        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        return x_tensor, y_tensor
