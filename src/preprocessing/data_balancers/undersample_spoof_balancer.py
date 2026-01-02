import numpy as np
import pandas as pd

from src.preprocessing.base_preprocessor import BasePreprocessor


class UndersampleSpoofBalancer(BasePreprocessor):
    def __init__(self, seed=42):
        super().__init__(__class__.__name__)
        self.seed = seed

    def transform(self, metadata: pd.DataFrame, embeddings: np.ndarray):
        spoof_mask = metadata["target"] == "spoof"
        metadata_spoof = metadata[spoof_mask]
        embeddings_spoof = embeddings[spoof_mask]

        bonafide_mask = ~spoof_mask
        metadata_bonafide = metadata[bonafide_mask]
        embeddings_bonafide = embeddings[bonafide_mask]

        bonafide_count = np.sum(bonafide_mask)
        if bonafide_count >= 0.5 * metadata.shape[0]:
            self.logger.info("Dataset is already balanced or bonafide samples are majority.")
            return metadata, embeddings

        np.random.seed(self.seed)
        sampled_ids = np.random.choice(metadata_spoof.shape[0], size=bonafide_count, replace=False)
        sampled_spoof_meta = metadata_spoof.iloc[sampled_ids]
        sampled_spoof_emb = embeddings_spoof[sampled_ids]

        new_metadata = pd.concat([metadata_bonafide, sampled_spoof_meta], ignore_index=True)
        new_embeddings = np.vstack((embeddings_bonafide, sampled_spoof_emb))

        shuffled_indices = np.random.permutation(new_metadata.shape[0])
        balanced_metadata = new_metadata.iloc[shuffled_indices].reset_index(drop=True)
        balanced_embeddings = new_embeddings[shuffled_indices]

        return balanced_metadata, balanced_embeddings
