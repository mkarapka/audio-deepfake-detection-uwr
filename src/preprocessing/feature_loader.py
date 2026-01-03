from pathlib import Path

import numpy as np
import pandas as pd

from src.common.constants import Constants as consts
from src.preprocessing.base_preprocessor import BasePreprocessor


class FeatureLoader(BasePreprocessor):
    def __init__(
        self,
        emb_file_name: str = consts.embeddings_file,
        data_dir: Path = consts.collected_data_dir,
    ):
        super().__init__(class_name=__class__.__name__)
        self.data_dir = data_dir
        self.emb_path = data_dir / emb_file_name

    def _get_file_path(self, file_name: str) -> Path:
        return self.data_dir / (file_name + consts.metadata_extension)

    def transform(self, file_name: str) -> tuple[pd.DataFrame, np.ndarray]:
        self.logger.info(f"Loading features from {file_name}.csv")
        file_path = self._get_file_path(file_name)

        loaded_meta = pd.read_csv(file_path, index_col=0)
        embeddings_mmap = np.load(self.emb_path, mmap_mode="r")
        loaded_embeddings = embeddings_mmap[loaded_meta.index].copy()

        return loaded_meta, loaded_embeddings
