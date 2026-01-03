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
            return metadata, embeddings
        (meta_bonafide, emb_bonafide), (meta_spoof, emb_spoof) = self.get_bonafide_spoof_data(metadata, embeddings)

        np.random.seed(self.seed)
        sampled_ids = self._gen_sampled_ids(
            spoof_samples_no=meta_spoof.shape[0],
            bonafide_samples_no=meta_bonafide.shape[0],
        )
        sampled_spoof_meta = meta_spoof.iloc[sampled_ids]
        sampled_spoof_emb = emb_spoof[sampled_ids]

        new_meta, new_emb = self.concat_data(
            l_meta=meta_bonafide,
            r_meta=sampled_spoof_meta,
            l_emb=emb_bonafide,
            r_emb=sampled_spoof_emb,
        )

        balanced_metadata, balanced_embeddings = self.shuffle_data(new_meta, new_emb)
        return balanced_metadata, balanced_embeddings
