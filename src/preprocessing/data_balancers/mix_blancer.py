import numpy as np
import pandas as pd

from src.preprocessing.data_balancers.base_balancer import BaseBalancer
from src.preprocessing.data_balancers.oversample_real_balancer import (
    OversampleRealBalancer,
)
from src.preprocessing.data_balancers.undersample_spoof_balancer import (
    UndersampleSpoofBalancer,
)


class MixBalancer(BaseBalancer):
    def __init__(self, seed=42, undersample_ratio=0.75, oversample_ratio=1.0):
        super().__init__(__class__.__name__)
        self.seed = seed
        self.undersample_ratio = undersample_ratio
        self.oversample_ratio = oversample_ratio
        self.undersampler = UndersampleSpoofBalancer(seed=seed, real_to_spoof_ratio=undersample_ratio)
        self.oversampler = OversampleRealBalancer(seed=seed, real_to_spoof_ratio=oversample_ratio)

    def transform(self, metadata: pd.DataFrame, features: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
        meta_under, features_under = self.undersampler.transform(metadata, features=features)
        balanced_metadata, balanced_features = self.oversampler.transform(meta_under, features=features_under)

        self.logger.info("MixBalancer - after undersampling and oversampling:")
        self.logger.info(f"{metadata.shape} -> {balanced_metadata.shape}")

        return balanced_metadata, balanced_features
