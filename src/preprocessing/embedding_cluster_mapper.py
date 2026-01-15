import pickle

import numpy as np
from hdbscan import HDBSCAN
from sklearn.decomposition import PCA
from src.common.basic_functions import setup_logger
from src.common.constants import Constants as consts
from src.preprocessing.base_preprocessor import BasePreprocessor


class EmbeddingClusterMapper(BasePreprocessor):
    def __init__(
        self, min_cluster_size: int = 1000, min_samples: int = 10, metric: str = "euclidean", random_state: int = 42
    ):
        super().__init__(class_name=__class__.__name__)
        self.logger = setup_logger(self.class_name, log_to_console=True)
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.random_state = random_state

    def _reduce_dimensions_PCA(self, embeddings, n_components):
        pca = PCA(n_components=n_components)
        return pca, pca.fit_transform(embeddings)

    def train(self, embeddings: np.ndarray, n_components=10):
        pca, reduced_embeddings = self._reduce_dimensions_PCA(embeddings, n_components=n_components)
        self.logger.info(f"Reduced embeddings shape after PCA: {reduced_embeddings.shape}")

        hdbscan = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_method="eom",
            prediction_data=True,
        )
        hdbscan.fit(reduced_embeddings)

        return hdbscan, pca

    def save_model(self, model, pca, file_name: str):
        file_path = consts.collected_data_dir / file_name
        with open(file_path, "wb") as f:
            pickle.dump((model, pca), f)

    def load_model(self, file_name: str):
        file_path = consts.collected_data_dir / file_name
        with open(file_path, "rb") as f:
            model, pca = pickle.load(f)
        return model, pca

    def transform(self, metadata, embeddings, pca, model):
        reduced_embeddings = pca.transform(embeddings)
        clusters = model.predict(reduced_embeddings)
        metadata["cluster"] = clusters
        return metadata
