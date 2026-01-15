import numpy as np

from src.preprocessing.embedding_cluster_mapper import EmbeddingClusterMapper
from src.preprocessing.feature_loader import FeatureLoader
from sklearn.preprocessing import Normalizer

class MapClustersIDsPipeline:
    def train_mapper(self, train_embeddings: np.ndarray):
        mapper = EmbeddingClusterMapper(min_cluster_size=100, min_samples=50, random_state=42)
        mapper.logger.info("Normalizing embeddings...")
        normalizer = Normalizer("l2")
        normalized_embeddings = normalizer.fit_transform(train_embeddings)
        mapper.logger.info("training cluster mapping model...")
        hdbscan, pca = mapper.train(normalized_embeddings, n_components=10)
        mapper.logger.info("training completed.")
        return hdbscan, pca

    def print_cluster_info(self, model, pca, embeddings: np.ndarray):
        mapper = EmbeddingClusterMapper()
        normalizer = Normalizer("l2")
        normalized_embeddings = normalizer.fit_transform(embeddings)
        clusters = mapper.transform(normalized_embeddings, pca=pca, model=model)
        unique, counts = np.unique(clusters, return_counts=True)
        cluster_info = dict(zip(unique, counts))
        mapper.logger.info(f"Cluster distribution: {cluster_info}")
        outliers = cluster_info.get(-1, 0)
        mapper.logger.info(f"Number of outliers: {outliers} out of {len(embeddings)} samples.")


if __name__ == "__main__":
    feature_loader = FeatureLoader(file_name="feature_extracted")
    train_meta = feature_loader.load_split_file(split_name="train")
    reduced_train_meta = feature_loader.sample_fraction(train_meta, fraction=0.04)
    train_embeddings = feature_loader.load_embeddings_from_metadata(reduced_train_meta)

    pipeline = MapClustersIDsPipeline()
    model, pca = pipeline.train_mapper(train_embeddings)
    pipeline.print_cluster_info(model, pca, train_embeddings)
