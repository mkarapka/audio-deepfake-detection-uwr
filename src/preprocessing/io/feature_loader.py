from pathlib import Path

import numpy as np
import pandas as pd

from src.common.constants import Constants as consts
from src.preprocessing.io.base_io import BaseIO


class FeatureLoader(BaseIO):
    def __init__(
        self,
        file_name=consts.feature_extracted,
        feat_suffix: str = "",
        data_dir: Path = consts.collected_data_dir,
        split_dir: Path = consts.split_dir,
    ):
        super().__init__(self.__class__.__name__, file_name, feat_suffix, data_dir, split_dir)
        self.emb_path = self._create_file_path(file_ext=consts.npy_ext)
        self.meta_path = self._create_file_path(file_ext=consts.csv_ext)

    def _sample_meta(self, metadata: pd.DataFrame, fraction=0.4) -> pd.DataFrame:
        sample_size = int(len(metadata) * fraction)
        reduced_split = metadata.sample(n=sample_size, random_state=42)
        return reduced_split

    def _sample_uq_audio_ids(self, metadata: pd.DataFrame, fraction: int, random_state=42):
        audio_ids = metadata["audio_id"].unique()

        np.random.seed(random_state)
        np.random.shuffle(audio_ids)

        sample_size = int(len(audio_ids) * fraction)
        return audio_ids[:sample_size]

    def sample_data(
        self, metadata: pd.DataFrame, embeddings: np.ndarray | None = None, fraction=0.4
    ) -> tuple[pd.DataFrame, np.ndarray] | pd.DataFrame:
        sampled_metadata = self._sample_meta(metadata, fraction=fraction)
        if embeddings is None:
            return sampled_metadata

        sampled_embeddings = embeddings[sampled_metadata.index]
        sampled_metadata = sampled_metadata.reset_index(drop=True)

        return sampled_metadata, sampled_embeddings

    def sample_data_by_audio_id(
        self, metadata: pd.DataFrame, embeddings: np.ndarray | None = None, fraction=0.4
    ) -> tuple[pd.DataFrame, np.ndarray | None]:
        sampled_audio_ids = self._sample_uq_audio_ids(metadata, fraction=fraction)

        sampled_metadata = metadata[metadata["audio_id"].isin(sampled_audio_ids)]
        if embeddings is None:
            return sampled_metadata

        sampled_embeddings = embeddings[sampled_metadata.index]
        sampled_metadata = sampled_metadata.reset_index(drop=True)

        return sampled_metadata, sampled_embeddings

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

    def load_meta_split(self, split_name: str) -> pd.DataFrame:
        file_path = self.create_read_file_path(file_ext=consts.csv_ext, split_name=split_name)
        return self.load_metadata_file(file_path, index_col=0)

    def load_embeddings_from_metadata(self, metadata: pd.DataFrame) -> np.ndarray:
        embeddings_mmap = np.load(self.emb_path, mmap_mode="r")
        return embeddings_mmap[metadata.index].copy()

    def load_data_split(self, split_name: str, index_col: int | None = 0) -> tuple[pd.DataFrame, np.ndarray]:
        file_path = self.create_read_file_path(file_ext=consts.csv_ext, split_name=split_name)
        self.logger.info(f"Loading features from {file_path}")

        loaded_meta = self.load_metadata_file(file_path, index_col=index_col)
        loaded_embeddings = self.load_embeddings_from_metadata(loaded_meta)
        loaded_meta = loaded_meta.reset_index(drop=True)

        return loaded_meta, loaded_embeddings

    def load_data(self) -> tuple[pd.DataFrame, np.ndarray]:
        self.logger.info(f"Loading features from {self.meta_path}")

        loaded_meta = self.load_metadata_file(self.meta_path)
        loaded_embeddings = self.load_embeddings_from_metadata(loaded_meta)
        loaded_meta = loaded_meta.reset_index(drop=True)

        return loaded_meta, loaded_embeddings
