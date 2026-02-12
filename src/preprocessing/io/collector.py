import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.constants import Constants as consts
from src.preprocessing.io.base_io import BaseIO


class Collector(BaseIO):
    def __init__(
        self,
        save_file_name: str,
        data_dir: Path = consts.collected_data_dir,
        split_dir: Path = consts.split_dir,
    ):
        super().__init__(
            class_name=__class__.__name__,
            file_name=save_file_name,
            feat_suffix="",
            data_dir=data_dir,
            split_dir=split_dir,
        )

        self.meta_data_file_path = self._create_file_path(file_ext=consts.csv_ext)
        self.embeddings_file_path = self._create_file_path(file_ext=consts.npy_ext)
        logging.info(f"Metadata will be saved to: {self.meta_data_file_path}")
        logging.info(f"Embeddings will be saved to: {self.embeddings_file_path}")

        if self.data_dir.exists() is False:
            self.data_dir.mkdir(parents=True, exist_ok=True)
        if self.split_dir.exists() is False:
            self.split_dir.mkdir(parents=True, exist_ok=True)

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

    def write_data_to_csv(self, data: pd.DataFrame, file_path: Path = None, include_index: bool = False):
        if file_path is None:
            file_path = self.meta_data_file_path
        self.logger.info(f"Saving metadata to {file_path}")
        if file_path.exists() is True:
            data.to_csv(file_path, mode="a+", index=include_index, header=False)
        else:
            data.to_csv(file_path, index=include_index, header=True)

    def transform(self, meta_df: pd.DataFrame, embeddings: np.ndarray):
        self.write_data_to_csv(meta_df)
        self._write_embeddings_to_npy(embeddings)

    def transform_splits(self, data: list[pd.DataFrame], splits=["train", "dev", "test"]):
        for meta, split_name in zip(data, splits):
            new_meta_path = self._create_file_path(file_ext=consts.csv_ext, split_name=split_name)
            self.write_data_to_csv(data=meta, file_path=new_meta_path, include_index=True)
