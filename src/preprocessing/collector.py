import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.constants import Constants as consts
from src.preprocessing.base_preprocessor import BasePreprocessor


class Collector(BasePreprocessor):
    def __init__(self, save_file_name: str):
        super().__init__(class_name=__class__.__name__)
        self.data_dir = consts.data_dir / "collected_data"
        if self.data_dir.exists() is False:
            self.data_dir.mkdir(parents=True, exist_ok=True)

        if save_file_name is not None:
            self.meta_data_file_path = self.data_dir / Path(save_file_name + consts.metadata_ext)
            self.embeddings_file_path = self.data_dir / Path(save_file_name + consts.embeddings_ext)
            logging.info(f"Metadata will be saved to: {self.meta_data_file_path}")
            logging.info(f"Embeddings will be saved to: {self.embeddings_file_path}")
        else:
            logging.error("Save file name must be provided for Collector.")

    def _write_data_to_csv(self, data: pd.DataFrame):
        if self.meta_data_file_path.exists() is True:
            data.to_csv(self.meta_data_file_path, mode="a+", index=False, header=False)
        else:
            data.to_csv(self.meta_data_file_path, index=False, header=True)

    def _write_embeddings_to_npy(self, embeddings: np.ndarray):
        if self.embeddings_file_path.exists() is True:
            existing_embeddings = np.load(self.embeddings_file_path)
            combined_embeddings = np.vstack((existing_embeddings, embeddings))
            np.save(self.embeddings_file_path, combined_embeddings)
        else:
            np.save(self.embeddings_file_path, embeddings)

    def get_meta_data_file_path(self) -> Path:
        return self.meta_data_file_path

    def get_embeddings_file_path(self) -> Path:
        return self.embeddings_file_path

    def transform(self, meta_df: pd.DataFrame, embeddings: np.ndarray):
        self._write_data_to_csv(meta_df)
        self._write_embeddings_to_npy(embeddings)
