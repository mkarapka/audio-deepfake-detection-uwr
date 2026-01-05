from pathlib import Path

import numpy as np
import pandas as pd

from src.common.constants import Constants as consts
from src.preprocessing.base_preprocessor import BasePreprocessor


class FeatureLoader(BasePreprocessor):
    def __init__(
        self,
        file_name=consts.wavlm_file_name_prefix,
        data_dir: Path = consts.collected_data_dir,
        split_dir: Path = consts.splited_data_dir,
    ):
        super().__init__(class_name=__class__.__name__)
        self.data_dir = data_dir
        self.split_dir = split_dir
        self.emb_path = data_dir / (file_name + consts.embeddings_extension)
        self.file_name = file_name

    def _get_file_path(self, split_name: str) -> Path:
        file_path = self.split_dir / (self.file_name + "_" + split_name + consts.metadata_extension)
        if not file_path.exists():
            self.logger.error(f"File {file_path} does not exist.")
        return file_path

    def transform(self, split_name: str) -> tuple[pd.DataFrame, np.ndarray]:
        self.logger.info(f"Loading features from {split_name}.csv")
        file_path = self._get_file_path(split_name)

        loaded_meta = pd.read_csv(file_path, index_col=0)
        embeddings_mmap = np.load(self.emb_path, mmap_mode="r")
        loaded_embeddings = embeddings_mmap[loaded_meta.index].copy()

        return loaded_meta, loaded_embeddings
