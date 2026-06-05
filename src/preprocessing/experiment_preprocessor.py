import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.common.constants import Constants as consts
from src.common.experiment_configs import BalanceStrategy
from src.common.logger import raise_error_logger, setup_logger
from src.common.utils import get_device
from src.datasets.audio_dataset import AudioDataset
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
    def __init__(self, feat_suffix: str, load_file_name: str = consts.feature_extracted, device: str = None):
        self.logger = setup_logger(__class__.__name__, log_to_console=True)
        self.feat_suffix = feat_suffix
        self.feature_loader = FeatureLoader(file_name=load_file_name, feat_suffix=feat_suffix)
        self.device = get_device(include_mps=True) if device is None else device
        self.logger.info(
            f"ExperimentPreprocessor initialized with device: {
                self.device} and feature suffix: '{feat_suffix}'"
        )

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

    def compute_standardize_params(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std[std == 0] = 1e-8
        return mean, std

    def compute_standardize_params_from_split(
        self, file_name: str = consts.feature_extracted, split_name: str = "train"
    ) -> tuple[np.ndarray, np.ndarray]:
        loader = FeatureLoader(file_name=file_name, feat_suffix=self.feat_suffix)
        meta, feat = loader.load_data_split(split_name=split_name)
        self.logger.info(
            f"Computed standardization params from '{file_name}' split '{split_name}' ({len(meta):,} samples)."
        )
        return self.compute_standardize_params(feat)

    def preprocess_data(
        self,
        splits_names: list[str],
        fraction: float | dict[str, float],
        use_audio_id_sampling: bool = False,
        use_standardize: bool = False,
        balance_splits_strategy: BalanceStrategy = None,
        remove_by_query: str | dict[str, str] | None = None,
        standardize_params: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> dict[str, AudioDataset]:
        split_dataset_dict = {}
        train_mean = standardize_params[0] if standardize_params else None
        train_std = standardize_params[1] if standardize_params else None

        for i, split_name in enumerate(splits_names):
            if self.feature_loader.file_name_no_suffix == consts.feature_extracted:
                meta, feat = self.feature_loader.load_data_split(split_name=split_name)
            else:
                meta, feat = self.feature_loader.load_data()

            if remove_by_query is not None:
                if isinstance(remove_by_query, dict):
                    query = remove_by_query[split_name]
                else:
                    query = remove_by_query
                self.logger.info(f"Removing records from split '{split_name}' using query: {query}")
                self.logger.info(f"Number of records before removal: {len(meta):,}")
                meta, feat = self._remove_records_by_query(metadata=meta, features=feat, query=query)
                self.logger.info(f"Number of records after removal: {len(meta):,}")

            if isinstance(fraction, dict):
                split_fraction = fraction[split_name]
            else:
                split_fraction = fraction

            if split_fraction < 1.0:
                self.logger.info(f"Sampling {split_fraction * 100:.1f}% of data for split '{split_name}'...")
                meta, feat = self.feature_loader.sample_data(
                    metadata=meta,
                    features=feat,
                    fraction=split_fraction,
                    audio_id_sampling=use_audio_id_sampling,
                )
                self.logger.info(f"Number of records after sampling: {len(meta):,}")

            if balance_splits_strategy is not None and balance_splits_strategy.get(split_name) is not None:
                self.logger.info(
                    f"Applying balancing strategy '{
                        balance_splits_strategy[split_name]}' to split '{split_name}'"
                )
                balance_type, ratio_args = balance_splits_strategy[split_name]
                balancer = self._get_balancer_instance(balancer_type=balance_type, ratio_args=ratio_args)
                if balancer is not None:
                    meta, feat = balancer.transform(metadata=meta, features=feat)
                self.logger.info(f"Number of records after balancing: {len(meta):,}")

            if use_standardize:
                self.logger.info(f"Standardizing features for split '{split_name}'...")
                if split_name == "train" and train_mean is None:
                    train_mean, train_std = self.compute_standardize_params(feat)

                torch_dataset = AudioDataset(
                    metadata=meta,
                    features=feat,
                    transform=lambda x: self._standardize_func(x, train_mean, train_std),
                )
            else:
                torch_dataset = AudioDataset(metadata=meta, features=feat)

            split_dataset_dict[split_name] = torch_dataset

        return split_dataset_dict

    def get_dataloaders(
        self, dataset_dict: dict[str, AudioDataset], batch_size: int, shuffle_train: bool = True
    ) -> dict[str, DataLoader]:
        dataloader_dict = {}
        for split_name, dataset in dataset_dict.items():
            shuffle = shuffle_train if split_name == "train" else False
            if self.device == "cuda" and dataset.device == "cpu":
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
            else:
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

            dataloader_dict[split_name] = dataloader

        return dataloader_dict
