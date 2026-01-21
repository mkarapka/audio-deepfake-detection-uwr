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
        self.meta_path = data_dir / (file_name + consts.metadata_extension)
        self.file_name = file_name

    def _get_file_path(self, split_name: str) -> Path:
        file_path = self.split_dir / (self.file_name + "_" + split_name + consts.metadata_extension)
        self.logger.info(f"Constructed file path: {file_path}")
        if not file_path.exists():
            self.logger.error(f"File {file_path} does not exist.")
        return file_path

    def load_speakers_ids(self) -> pd.DataFrame:
        speakers_ids_path = self.data_dir / consts.speakers_ids_file
        self.logger.info(f"Loading speakers IDs from {speakers_ids_path}")
        speakers_ids_df = pd.read_csv(speakers_ids_path)
        return speakers_ids_df

    def load_metadata_file(self, file_path: Path | None = None, index_col: int | None = None) -> pd.DataFrame:
        if file_path is None:
            file_path = self.meta_path
        self.logger.info(f"Loading metadata from {file_path}")
        metadata_df = pd.read_csv(file_path, index_col=index_col)
        return metadata_df

    def load_embeddings_from_metadata(self, metadata: pd.DataFrame) -> np.ndarray:
        embeddings_mmap = np.load(self.emb_path, mmap_mode="r")
        return embeddings_mmap[metadata.index].copy()

    def load_split_file(self, split_name: str) -> pd.DataFrame:
        file_path = self._get_file_path(split_name)
        return self.load_metadata_file(file_path, index_col=0)

    def sample_fraction(self, metadata: pd.DataFrame, fraction=0.4) -> pd.DataFrame:
        sample_size = int(len(metadata) * fraction)
        reduced_split = metadata.sample(n=sample_size, random_state=42)
        return reduced_split

    def sample_fraction_metadata_and_embeddings(
        self, metadata: pd.DataFrame, embeddings: np.ndarray, fraction=0.4
    ) -> tuple[pd.DataFrame, np.ndarray]:
        sampled_metadata = self.sample_fraction(metadata, fraction=fraction)
        sampled_embeddings = embeddings[sampled_metadata.index]
        sampled_metadata = sampled_metadata.reset_index(drop=True)
        return sampled_metadata, sampled_embeddings

    def transform(self, split_name: str, index_col: int | None = 0) -> tuple[pd.DataFrame, np.ndarray]:
        file_path = self._get_file_path(split_name)
        self.logger.info(f"Loading features from {file_path}")

        loaded_meta = self.load_metadata_file(file_path, index_col=index_col)
        loaded_embeddings = self.load_embeddings_from_metadata(loaded_meta)
        loaded_meta = loaded_meta.reset_index(drop=True)

        return loaded_meta, loaded_embeddings

    def transfrorm_all(self) -> tuple[pd.DataFrame, np.ndarray]:
        self.logger.info(f"Loading features from {self.meta_path}")

        loaded_meta = self.load_metadata_file(self.meta_path)
        loaded_embeddings = self.load_embeddings_from_metadata(loaded_meta)
        loaded_meta = loaded_meta.reset_index(drop=True)

        return loaded_meta, loaded_embeddings
