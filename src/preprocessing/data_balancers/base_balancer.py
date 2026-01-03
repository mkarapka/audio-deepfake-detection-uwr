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

    def shuffle_data(self, metadata: pd.DataFrame):
        shuffled_indices = np.random.permutation(metadata.shape[0])
        balanced_metadata = metadata.iloc[shuffled_indices]
        return balanced_metadata

    def get_bonafide_spoof_data(self, metadata: pd.DataFrame):
        spoof_mask = metadata["target"] == "spoof"
        bonafide_mask = ~spoof_mask

        return metadata[bonafide_mask], metadata[spoof_mask]
