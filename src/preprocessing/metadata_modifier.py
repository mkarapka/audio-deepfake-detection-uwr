import pandas as pd

from src.preprocessing.base_preprocessor import BasePreprocessor


class MetadataModifier(BasePreprocessor):
    def __init__(self, audio_type: str, speakers_ids: pd.DataFrame):
        super().__init__(__class__.__name__)
        if audio_type is None:
            self.logger.error("Current audio type must be provided to initialize MetadataModifier.")
        self.audio_type = audio_type

        if speakers_ids is None:
            self.logger.error("Speakers IDs DataFrame must be provided to initialize MetadataModifier.")
        self.speakers_ids = speakers_ids

    def _get_split_and_record_id(self, key: str):
        split_end_idx = key.find("/")
        split = key[:split_end_idx]

        record_id_end_idx = key.find("_", split_end_idx)
        if record_id_end_idx == -1:
            return split, int(key[split_end_idx + 1 :])
        return split, int(key[split_end_idx + 1 : record_id_end_idx])

    def _set_speaker_id_for_each_record(self, metadata: pd.DataFrame):
        return metadata.apply(lambda row: self.speakers_ids.loc[row["record_id"], row["split"]], axis=1)

    def transform(self, current_config: str, metadata: pd.DataFrame):
        splits, record_ids = [], []
        for key in metadata["key_id"]:
            split, rec_id = self._get_split_and_record_id(key)
            splits.append(split)
            record_ids.append(rec_id)

        metadata["config"] = current_config
        metadata["split"] = splits
        metadata["record_id"] = record_ids
        metadata["speaker_id"] = self._set_speaker_id_for_each_record(metadata)
        metadata["target"] = self.audio_type
        metadata = metadata.drop(columns=["key_id"])

        return metadata
