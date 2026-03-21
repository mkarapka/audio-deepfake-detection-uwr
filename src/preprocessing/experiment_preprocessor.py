import numpy as np
import pandas as pd

from src.common.logger import raise_error_logger, setup_logger
from src.models.model_trainer import ModelTrainer
from src.preprocessing.data_balancers.base_balancer import BaseBalancer
from src.preprocessing.data_balancers.mix_balancer import MixBalancer
from src.preprocessing.data_balancers.oversample_real_balancer import (
    OversampleRealBalancer,
)
from src.preprocessing.data_balancers.undersample_spoof_balancer import (
    UndersampleSpoofBalancer,
)
from src.preprocessing.io.collector import Collector
from src.preprocessing.io.feature_loader import FeatureLoader


class ExperimentPreprocessor:
    def __init__(self, load_file_name: str, save_file_name: str, feat_suffix: str):
        self.logger = setup_logger(__class__.__name__, log_to_console=True)
        self.feature_loader = FeatureLoader(file_name=load_file_name, feat_suffix=feat_suffix)
        self.collector = Collector(save_file_name=save_file_name, feat_suffix=feat_suffix)
        self.results = {}
        self.trainer = ModelTrainer()

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

    def _sample_data(
        self,
        metadata: pd.DataFrame,
        features: np.ndarray,
        fraction: float,
        is_audio_ids_sampling: bool,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        if is_audio_ids_sampling:
            return self.feature_loader.sample_data(metadata=metadata, features=features, fraction=fraction)
        return self.feature_loader.sample_by_audio_ids(metadata=metadata, features=features, fraction=fraction)

    def _convert_labels_to_ints(self, y: pd.Series, pos_label: str) -> np.ndarray:
        return (y == pos_label).astype(int)

    def preprocess_data(
        self,
        splits_names: list[str],
        fraction: float,
        is_audio_ids_sampling: bool,
        balance_splits_strategy: tuple[str, float | list[float]],
    ) -> dict[str, tuple[pd.DataFrame, np.ndarray]]:

        data_for_exp = {}
        for split_name in splits_names:
            meta, feat = self.feature_loader.load_data_split(split_name=split_name)

            if fraction < 1.0:
                meta, feat = self._sample_data(
                    metadata=meta,
                    features=feat,
                    fraction=fraction,
                    is_audio_ids_sampling=is_audio_ids_sampling,
                )

            if balance_splits_strategy is not None:
                balance_type, ratio_args = balance_splits_strategy
                balancer = self._get_balancer_instance(balancer_type=balance_type, ratio_args=ratio_args)
                if balancer is not None:
                    meta, feat = balancer.transform(metadata=meta, features=feat)

            data_for_exp[split_name] = (meta, feat)

        return data_for_exp

    def get_target(self, metadata: pd.DataFrame, pos_label="bonafide") -> np.ndarray:
        y = self._convert_labels_to_ints(metadata["target"], pos_label=pos_label)
        return y

    def get_X_y(self, metadata: pd.DataFrame, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y = self.get_target(metadata=metadata)
        return features, y

    def remove_subclass_from_split(
        self, metadata: pd.DataFrame, features: np.ndarray, subclass_label: int
    ) -> tuple[pd.DataFrame, np.ndarray]:
        pass
