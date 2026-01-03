import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing.base_preprocessor import BasePreprocessor


class DatasetSpliter(BasePreprocessor):
    def __init__(
        self,
        config: dict[str, float],
        speakers_ids: pd.DataFrame,
        seed: int | None,
    ):
        super().__init__()
        if config is None:
            self.logger.error("Config must be provided to initialize DatasetSpliter.")
        self.config = config

        if speakers_ids is None:
            self.logger.error("Speakers IDs DataFrame must be provided to initialize DatasetSpliter.")
        self.speakers_ids = speakers_ids
        self.seed = seed

    def _get_unique_speakers_ids(self) -> np.ndarray:
        dev_ids = self.speakers_ids["dev"].unique()
        test_ids = self.speakers_ids["test"][self.speakers_ids["test"] != -1].unique()
        return np.hstack([dev_ids, test_ids])

    def _get_train_dev_test_speakers_ids(self, uq_speakers_ids: np.ndarray):
        ids_train, ids_dev_test = train_test_split(
            uq_speakers_ids,
            test_size=(self.config["dev"] + self.config["test"]),
            random_state=self.seed,
        )
        test_ratio = self.config["test"] / (self.config["dev"] + self.config["test"])
        ids_dev, ids_test = train_test_split(ids_dev_test, test_size=test_ratio, random_state=self.seed)
        return ids_train, ids_dev, ids_test

    def transform(self, metadata: pd.DataFrame):
        def create_split_mask(split_ids):
            return metadata["speaker_id"].isin(split_ids)

        uq_speaker_ids = self._get_unique_speakers_ids()
        ids_train, ids_dev, ids_test = self._get_train_dev_test_speakers_ids(uq_speaker_ids)

        mask_train = create_split_mask(ids_train)
        mask_dev = create_split_mask(ids_dev)
        mask_test = create_split_mask(ids_test)

        return metadata[mask_train], metadata[mask_dev], metadata[mask_test]
