import numpy as np
import pandas as pd

from src.preprocessing.data_balancers.base_balancer import BaseBalancer


class UndersampleSpoofBalancer(BaseBalancer):
    def __init__(self, seed=42, real_to_spoof_ratio=1.0):
        super().__init__(__class__.__name__)
        self.seed = seed
        self.real_to_spoof_ratio = real_to_spoof_ratio

    def _gen_sampled_ids(self, spoof_samples_no: int, bonafide_samples_no: int):
        adjusted_spoof_count = int(bonafide_samples_no / self.real_to_spoof_ratio)
        if adjusted_spoof_count >= spoof_samples_no:
            self.logger.info("Adjusted spoof count exceeds available spoof samples. Using all spoof samples.")
            return np.arange(spoof_samples_no)
        return np.random.choice(spoof_samples_no, size=adjusted_spoof_count, replace=False)

    def transform(self, metadata: pd.DataFrame, embeddings: np.ndarray):
        if not self.is_need_to_balance(metadata):
            return metadata
        meta_bonafide, meta_spoof = self.get_bonafide_spoof_data(metadata)

        np.random.seed(self.seed)
        sampled_ids = self._gen_sampled_ids(
            spoof_samples_no=meta_spoof.shape[0],
            bonafide_samples_no=meta_bonafide.shape[0],
        )
        sampled_spoof_meta = meta_spoof.iloc[sampled_ids]

        new_meta = pd.concat((meta_bonafide, sampled_spoof_meta))
        balanced_metadata = self.shuffle_data(new_meta)
        return balanced_metadata, embeddings[balanced_metadata.index]
