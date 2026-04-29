import numpy as np
import pandas as pd

from src.common.constants import Constants as consts
from src.common.logger import raise_error_logger, setup_logger
from src.models.torch_data_loader import AudioDataset
from src.preprocessing.data_balancers.base_balancer import BaseBalancer
from src.preprocessing.data_balancers.mix_balancer import MixBalancer
from src.preprocessing.data_balancers.oversample_real_balancer import (
    OversampleRealBalancer,
)
from src.preprocessing.data_balancers.undersample_spoof_balancer import (
    UndersampleSpoofBalancer,
)
from src.preprocessing.io.feature_loader import FeatureLoader


class ExperimentPreprocessor:
    def __init__(self, feat_suffix: str, load_file_name: str = consts.feature_extracted):
        self.logger = setup_logger(__class__.__name__, log_to_console=True)
        self.feature_loader = FeatureLoader(file_name=load_file_name, feat_suffix=feat_suffix)

    def _get_balancer_instance(self, balancer_type: str, ratio_args: float | list[float]) -> BaseBalancer:
        if balancer_type == "undersample":
            return UndersampleSpoofBalancer(real_to_spoof_ratio=ratio_args)
        elif balancer_type == "oversample":
            return OversampleRealBalancer(real_to_spoof_ratio=ratio_args)
        elif balancer_type == "mix":
            undersample_ratio, oversample_ratio = ratio_args
            return MixBalancer(undersample_ratio=undersample_ratio, oversample_ratio=oversample_ratio)
        elif balancer_type == "unbalanced":
            return None
        else:
            raise_error_logger(self.logger, f"Unknown balancer type: {balancer_type}")

    def _remove_records_by_query(
        self, metadata: pd.DataFrame, features: np.ndarray, query: str
    ) -> tuple[pd.DataFrame, np.ndarray]:
        mask = np.logical_not(metadata.eval(query))
        return metadata[mask].reset_index(drop=True), features[mask]

    def _standardize_func(self, x, train_mean, train_std):
        if train_mean is None or train_std is None:
            raise_error_logger(
                self.logger,
                "Standardization is enabled but train split is not processed yet.",
            )
        return (x - train_mean) / train_std

    def preprocess_data(
        self,
        splits_names: list[str],
        fraction: float,
        use_audio_id_sampling: bool = False,
        use_standardize: bool = False,
        balance_splits_strategy: list[tuple[str, float | list[float] | None]] = None,
        remove_by_query: str = None,
        device: str = None,
    ) -> dict[str, AudioDataset]:
        split_dataset_dict = {}
        train_mean = None
        train_std = None

        for i, split_name in enumerate(splits_names):
            meta, feat = self.feature_loader.load_data_split(split_name=split_name)

            if remove_by_query is not None:
                meta, feat = self._remove_records_by_query(metadata=meta, features=feat, query=remove_by_query)

            if fraction < 1.0:
                meta, feat = self.feature_loader.sample_data(
                    metadata=meta,
                    features=feat,
                    fraction=fraction,
                    audio_id_sampling=use_audio_id_sampling,
                )

            if balance_splits_strategy is not None and balance_splits_strategy[i] is not None:
                balance_type, ratio_args = balance_splits_strategy[i]
                balancer = self._get_balancer_instance(balancer_type=balance_type, ratio_args=ratio_args)
                if balancer is not None:
                    meta, feat = balancer.transform(metadata=meta, features=feat)

            if use_standardize:
                if split_name == "train":
                    train_mean = np.mean(feat, axis=0)
                    train_std = np.std(feat, axis=0)
                    train_std[train_std == 0] = 1e-8

                torch_dataset = AudioDataset(
                    metadata=meta,
                    features=feat,
                    transform=lambda x: self._standardize_func(x, train_mean, train_std),
                    device=device,
                )
            else:
                torch_dataset = AudioDataset(metadata=meta, features=feat, device=device)

            split_dataset_dict[split_name] = torch_dataset

        return split_dataset_dict

    def get_target(self, metadata: pd.DataFrame, pos_label="bonafide") -> np.ndarray:
        return (metadata["target"] == pos_label).astype(int)

    def get_X_y(self, metadata: pd.DataFrame, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y = self.get_target(metadata=metadata)
        return features, y
