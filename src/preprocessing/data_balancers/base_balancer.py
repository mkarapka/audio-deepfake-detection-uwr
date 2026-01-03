import numpy as np
import pandas as pd

from src.preprocessing.base_preprocessor import BasePreprocessor


class BaseBalancer(BasePreprocessor):
    def __init__(self, name: str):
        super().__init__(name)

    def is_need_to_balance(self, metadata: pd.DataFrame):
        bonafide_count = np.sum(metadata["target"] != "spoof")
        if bonafide_count >= 0.5 * metadata.shape[0]:
            self.logger.info("Dataset is already balanced or bonafide samples are majority.")
            return False
        return True

    def concat_data(
        self,
        l_meta: pd.DataFrame,
        r_meta: pd.DataFrame,
        l_emb: np.ndarray,
        r_emb: np.ndarray,
    ):
        new_metadata = pd.concat((l_meta, r_meta), ignore_index=True)
        new_embeddings = np.vstack((l_emb, r_emb))
        return new_metadata, new_embeddings

    def shuffle_data(self, metadata: pd.DataFrame, embeddings: np.ndarray):
        shuffled_indices = np.random.permutation(metadata.shape[0])
        balanced_metadata = metadata.iloc[shuffled_indices].reset_index(drop=True)
        balanced_embeddings = embeddings[shuffled_indices]
        return balanced_metadata, balanced_embeddings

    def get_bonafide_spoof_data(self, metadata: pd.DataFrame, embeddings: np.ndarray):
        spoof_mask = metadata["target"] == "spoof"
        metadata_spoof = metadata[spoof_mask]
        embeddings_spoof = embeddings[spoof_mask]

        bonafide_mask = ~spoof_mask
        metadata_bonafide = metadata[bonafide_mask]
        embeddings_bonafide = embeddings[bonafide_mask]

        return (metadata_bonafide, embeddings_bonafide), (metadata_spoof, embeddings_spoof)
