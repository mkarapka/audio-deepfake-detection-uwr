import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.constants import Constants as consts
from src.preprocessing.base_preprocessor import BasePreprocessor


class Collector(BasePreprocessor):
    def __init__(self, save_file_name: str, data_dir: Path | None = consts.collected_data_dir):
        super().__init__(class_name=__class__.__name__)
        self.data_dir = data_dir
        self.file_name = save_file_name
        self.meta_data_file_path = None
        self.embeddings_file_path = None

        if self.data_dir.exists() is False:
            self.data_dir.mkdir(parents=True, exist_ok=True)

        if save_file_name is not None:
            self.meta_data_file_path = self._create_file_path(f_name_suffix=consts.metadata_extension)
            self.embeddings_file_path = self._create_file_path(f_name_suffix=consts.embeddings_extension)
            logging.info(f"Metadata will be saved to: {self.meta_data_file_path}")
            logging.info(f"Embeddings will be saved to: {self.embeddings_file_path}")
        else:
            logging.error("Save file name must be provided for Collector.")

    def _create_file_path(self, f_name_suffix: str):
        file_path = self.data_dir / Path(f"{self.file_name}_{f_name_suffix}")
        return file_path

    def _write_data_to_csv(self, data: pd.DataFrame, file_path: Path = None, include_index: bool = False):
        if file_path is None:
            file_path = self.meta_data_file_path
        if file_path.exists() is True:
            data.to_csv(file_path, mode="a+", index=include_index, header=False)
        else:
            data.to_csv(file_path, index=include_index, header=True)

    def _write_embeddings_to_npy(self, embeddings: np.ndarray, file_path: Path = None):
        if file_path is None:
            file_path = self.embeddings_file_path
        if file_path.exists() is True:
            existing_embeddings = np.load(file_path)
            combined_embeddings = np.vstack((existing_embeddings, embeddings))
            np.save(file_path, combined_embeddings)
        else:
            np.save(file_path, embeddings)

    def get_metadata_file_path(self) -> Path:
        return self.meta_data_file_path

    def get_embeddings_file_path(self) -> Path:
        return self.embeddings_file_path

    def transform(self, meta_df: pd.DataFrame, embeddings: np.ndarray):
        self._write_data_to_csv(meta_df)
        self._write_embeddings_to_npy(embeddings)

    def transform_splits(self, data: list[pd.DataFrame], splits=["train", "dev", "test"]):
        for meta, split_name in zip(data, splits):
            new_meta_path = self._create_file_path(f"{split_name}{consts.metadata_extension}")
            self._write_data_to_csv(data=meta, file_path=new_meta_path, include_index=True)
