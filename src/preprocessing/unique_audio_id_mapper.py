import pandas as pd

from src.preprocessing.base_preprocessor import BasePreprocessor


class UniqueAudioIdMapper(BasePreprocessor):
    def __init__(self):
        self.counter = 0
        self.unique_audio_ids = {}

    def _create_uq_audio_id(self, config_id: str, split_id: str, record_id: int):
        uq_audio_id = f"{config_id}_{split_id}_{record_id}"
        return uq_audio_id

    def transform(self, metadata: pd.DataFrame):
        uq_audio_ids = []
        for _, row in metadata.iterrows():
            config_id = row["config"]
            split_id = row["split"]
            record_id = row["record_id"]

            uq_audio_id = self._create_uq_audio_id(config_id, split_id, record_id)
            if uq_audio_id not in self.unique_audio_ids:
                self.unique_audio_ids[uq_audio_id] = self.counter
                self.counter += 1
            uq_audio_ids.append(self.unique_audio_ids[uq_audio_id])

        metadata["unique_audio_id"] = uq_audio_ids
        return metadata
