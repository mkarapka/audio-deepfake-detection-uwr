from src.preprocessing.base_preprocessor import BasePreprocessor
import pandas as pd


class MetadataModifier(BasePreprocessor):
    def __init__(self, audio_type : str):
        super().__init__(__class__.__name__)
        if audio_type is None:
            self.logger.error("Current audio type must be provided to initialize MetadataModifier.")
        
        self.audio_type = audio_type

    def _get_split_and_record_id(self, key: str):
        split_end_idx = key.find("/")
        split = key[:split_end_idx]

        record_id_end_idx = key.find("_", split_end_idx)
        if record_id_end_idx == -1:
            return split, int(key[split_end_idx + 1 :])
        return split, int(key[split_end_idx + 1 : record_id_end_idx])

    def transform(self, current_config : str, metadata: pd.DataFrame):
        splits, record_ids = [], []
        for key in metadata["key_id"]:
            split, rec_id = self._get_split_and_record_id(key)
            splits.append(split)
            record_ids.append(rec_id)

        metadata["config"] = current_config
        metadata["split"] = splits
        metadata["record_id"] = record_ids
        metadata["target"] = self.audio_type
        
        metadata = metadata.drop(columns=["key_id"])
        return metadata
