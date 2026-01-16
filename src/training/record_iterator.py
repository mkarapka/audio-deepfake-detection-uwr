import numpy as np
import pandas as pd

from src.common.basic_functions import setup_logger


class RecordIterator:
    def __init__(self):
        self.logger = setup_logger(__class__.__name__, log_to_console=True)

    def sample_fraction(self, metadata: pd.DataFrame, fraction: float) -> pd.DataFrame:
        unique_records_ids = metadata["unique_audio_id"].unique()

        sample_size = int(len(unique_records_ids) * fraction)
        sampled_record_ids = np.random.choice(unique_records_ids, size=sample_size, replace=False)
        new_metadata = metadata[metadata["unique_audio_id"].isin(sampled_record_ids)]
        self.logger.info(f"Sampled {sample_size} unique records out of {len(unique_records_ids)}")
        self.logger.info(f"New metadata shape: {new_metadata.shape}, original shape: {metadata.shape}")
        self.logger.info(f"Fraction of records in new metadata: {new_metadata.shape[0] / metadata.shape[0]:.4f}")

        return new_metadata

    def iterate_records(self, metadata: pd.DataFrame, embeddings: np.ndarray):
        unique_records_ids = metadata["unique_audio_id"].unique()

        for unique_audio_id in unique_records_ids:
            mask = metadata["unique_audio_id"] == unique_audio_id
            record_embeddings = embeddings[mask]

            yield record_embeddings, mask
