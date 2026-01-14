import pickle
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.common.constants import Constants as consts
from src.preprocessing.base_preprocessor import BasePreprocessor


class EmbeddingClusterMapper(BasePreprocessor):
    def __init__(self, K_list: list[int], random_state=37):
        super().__init__(class_name=__class__.__name__)
        self.K_list = K_list
        self.random_state = random_state

    def _reduce_dimensions_PCA(self, embeddings, n_components=64):
        pca = PCA(n_components=n_components)
        return pca, pca.fit_transform(embeddings)

    def _reduce_dimentions_UMAP(self, embeddings):
        pass

    def get_info_for_each_cluster(self, embeddings):
        _, reduced_embeddings = self._reduce_dimensions_PCA(embeddings)

        slihouette_scores = {}
        intertias = {}
        for k in self.K_list:
            print(f"Evaluating for k={k}")
            k_means = KMeans(n_clusters=k, random_state=self.random_state)
            clusters = k_means.fit_predict(reduced_embeddings)
            silhouette_avg = silhouette_score(reduced_embeddings, clusters)
            intertia = k_means.inertia_

            slihouette_scores[k] = silhouette_avg
            intertias[k] = intertia

        return slihouette_scores, intertias

    def train(self, embeddings, n_clusters: int):
        pca, reduced_embeddings = self._reduce_dimensions_PCA(embeddings)

        k_means = KMeans(n_clusters=n_clusters, random_state=self.random_state)
        k_means.fit(reduced_embeddings)

        return k_means, pca

    def save_model(self, model, pca, file_name: Path):
        file_path = consts.collected_data_dir / file_name
        with open(file_path, "wb") as f:
            pickle.dump((model, pca), f)

    def load_model(self, file_name: Path):
        file_path = consts.collected_data_dir / file_name
        with open(file_path, "rb") as f:
            model, pca = pickle.load(f)
        return model, pca

    def transform(self, metadata, embeddings, pca, model):
        reduced_embeddings = pca.transform(embeddings)
        clusters = model.predict(reduced_embeddings)
        metadata["cluster"] = clusters
        return metadata
