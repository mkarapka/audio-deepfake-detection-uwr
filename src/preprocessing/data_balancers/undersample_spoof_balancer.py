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

    def _gen_sampled_bonafide_ids(self, bonafide_samples_no: int, spoof_samples_no: int):
        adjusted_bonafide_count = int(spoof_samples_no * self.real_to_spoof_ratio)
        if adjusted_bonafide_count >= bonafide_samples_no:
            self.logger.info("Adjusted bonafide count exceeds available bonafide samples. Using all bonafide samples.")
            return np.arange(bonafide_samples_no)
        return np.random.choice(bonafide_samples_no, size=adjusted_bonafide_count, replace=False)

    def transform(
        self, metadata: pd.DataFrame, features: np.ndarray, reduce_spoof: bool = False
    ) -> tuple[pd.DataFrame, np.ndarray]:
        meta_bonafide, meta_spoof = self.get_bonafide_spoof_data(metadata)
        bonafide_count = meta_bonafide.shape[0]
        spoof_count = meta_spoof.shape[0]

        if bonafide_count == 0 or spoof_count == 0:
            self.logger.info("Skipping undersampling because one of the classes is empty.")
            return metadata, features

        np.random.seed(self.seed)
        sampled_bonafide_ids = np.arange(bonafide_count)
        sampled_spoof_ids = np.arange(spoof_count)

        if reduce_spoof:
            self.logger.info("Reducing spoof samples based on bonafide samples count and real_to_spoof_ratio.")
            sampled_spoof_ids = self._gen_sampled_ids(
                spoof_samples_no=spoof_count,
                bonafide_samples_no=bonafide_count,
            )

            # If spoof is already minority, reducing spoof cannot improve the target ratio.
            current_ratio = bonafide_count / spoof_count
            if sampled_spoof_ids.shape[0] == spoof_count and current_ratio > self.real_to_spoof_ratio:
                self.logger.info("Spoof is already minority for the target ratio. Reducing bonafide samples instead.")
                sampled_bonafide_ids = self._gen_sampled_bonafide_ids(
                    bonafide_samples_no=bonafide_count,
                    spoof_samples_no=spoof_count,
                )
        else:
            current_ratio = bonafide_count / spoof_count
            if current_ratio < self.real_to_spoof_ratio:
                sampled_spoof_ids = self._gen_sampled_ids(
                    spoof_samples_no=spoof_count,
                    bonafide_samples_no=bonafide_count,
                )
            elif current_ratio > self.real_to_spoof_ratio:
                sampled_bonafide_ids = self._gen_sampled_bonafide_ids(
                    bonafide_samples_no=bonafide_count,
                    spoof_samples_no=spoof_count,
                )
            else:
                self.logger.info("Dataset already matches target ratio. Skipping undersampling.")
                return metadata, features

        sampled_bonafide_meta = meta_bonafide.iloc[sampled_bonafide_ids]
        sampled_spoof_meta = meta_spoof.iloc[sampled_spoof_ids]
        new_meta = pd.concat((sampled_bonafide_meta, sampled_spoof_meta))
        balanced_metadata = self.shuffle_data(new_meta)

        balanced_features = features[balanced_metadata.index]
        self.previous_index = balanced_metadata.index
        balanced_metadata = balanced_metadata.reset_index(drop=True)

        return balanced_metadata, balanced_features
