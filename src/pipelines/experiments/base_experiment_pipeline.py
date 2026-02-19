from numpy import ndarray
from pandas import DataFrame

from src.common.constants import BalanceType
from src.common.constants import Constants as consts
from src.common.constants import SplitConfig
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


class BaseExperimentPipeline:
    def __init__(
        self,
        file_name=consts.feature_extracted,
    ):
        self.logger = setup_logger(__class__.__name__, log_to_console=True)
        self.feature_loader = FeatureLoader(file_name=file_name, feat_suffix="")
        self.collector = Collector(save_file_name=file_name, feat_suffix="")
        self.trainer = ModelTrainer()

    def _get_balancer_instance(self, balancer_type: BalanceType, ratio_args: float | list[float]) -> BaseBalancer:
        if balancer_type == BalanceType.UNDERSAMPLE:
            return UndersampleSpoofBalancer(real_to_spoof_ratio=ratio_args)
        elif balancer_type == BalanceType.OVERSAMPLE:
            return OversampleRealBalancer(real_to_spoof_ratio=ratio_args)
        elif balancer_type == BalanceType.MIX:
            undersample_ratio, oversample_ratio = ratio_args
            return MixBalancer(undersample_ratio=undersample_ratio, oversample_ratio=oversample_ratio)
        elif balancer_type == BalanceType.UNBALANCED:
            return None
        else:
            raise_error_logger(self.logger, f"Unknown balancer type: {balancer_type}")

    def _sample_data(
        self,
        metadata: DataFrame,
        features: ndarray,
        fraction: float,
        is_audio_ids_sampling: bool,
    ) -> tuple[DataFrame, ndarray]:
        if is_audio_ids_sampling:
            return self.feature_loader.sample_data(metadata=metadata, features=features, fraction=fraction)
        return self.feature_loader.sample_by_audio_ids(metadata=metadata, features=features, fraction=fraction)

    def preprocess_data(
        self, splits_config: dict[str, SplitConfig], fraction: float, is_audio_ids_sampling: bool
    ) -> dict[str, tuple[DataFrame, ndarray]]:
        data_for_exp = {}
        for split_name, config in splits_config.items():
            meta, feat = self.feature_loader.load_data_split(split_name=split_name)
            sampled_meta, sampled_feat = self._sample_data(
                metadata=meta,
                features=feat,
                fraction=fraction,
                is_audio_ids_sampling=is_audio_ids_sampling,
            )

            balancer = self._get_balancer_instance(config.balance_type, config.ratio_args)
            if balancer is not None:
                balanced_meta, balanced_feat = balancer.transform(metadata=sampled_meta, features=sampled_feat)
            data_for_exp[split_name] = (balanced_meta, balanced_feat)

        return data_for_exp

    def create_params_dictionary(self):
        raise_error_logger(self.logger, "Subclasses should implement this method.", error_type=NotImplementedError)

    def run(self):
        raise_error_logger(self.logger, "Subclasses should implement this method.", error_type=NotImplementedError)
