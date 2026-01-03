import numpy as np
import pandas as pd

from src.preprocessing.data_balancers.base_balancer import BaseBalancer


class OversampleRealBalancer(BaseBalancer):
    def __init__(self, seed=42, real_to_spoof_ratio=1.0):
        super().__init__(__class__.__name__)
        self.seed = seed
        self.real_to_spoof_ratio = real_to_spoof_ratio

    def _gen_sampled_ids(self, total_spoof_samples: int, bonafide_samples: int):
        adjusted_bonafide_count = int(total_spoof_samples * self.real_to_spoof_ratio) - bonafide_samples
        if adjusted_bonafide_count <= 0:
            self.logger.info("No need to oversample bonafide samples. New adjusted bonafide count is non-positive.")
            return np.array([], dtype=int)
        return np.random.choice(bonafide_samples, size=adjusted_bonafide_count, replace=True)

    def transform(self, metadata: pd.DataFrame, embeddings: np.ndarray):
        if not self.is_need_to_balance(metadata):
            return metadata, embeddings
        (meta_bonafide, emb_bonafide), (meta_spoof, _) = self.get_bonafide_spoof_data(metadata, embeddings)

        np.random.seed(self.seed)
        sampled_ids = self._gen_sampled_ids(
            total_spoof_samples=meta_spoof.shape[0],
            bonafide_samples=meta_bonafide.shape[0],
        )
        sampled_bonafide_meta = meta_bonafide.iloc[sampled_ids]
        sampled_bonafide_emb = emb_bonafide[sampled_ids]

        new_meta, new_emb = self.concat_data(
            l_meta=metadata, r_meta=sampled_bonafide_meta, l_emb=embeddings, r_emb=sampled_bonafide_emb
        )
        balanced_metadata, balanced_embeddings = self.shuffle_data(new_meta, new_emb)
        return balanced_metadata, balanced_embeddings
