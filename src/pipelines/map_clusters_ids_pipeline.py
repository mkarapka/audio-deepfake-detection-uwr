import numpy as np

from src.preprocessing.embedding_cluster_mapper import EmbeddingClusterMapper
from src.preprocessing.feature_loader import FeatureLoader


class MapClustersIDsPipeline:
    def train_mapper(self, train_embeddings: np.ndarray):
        mapper = EmbeddingClusterMapper(min_cluster_size=100, min_samples=5, random_state=42)
        mapper.logger.info("training cluster mapping model...")
        mapper.train(train_embeddings)
        mapper.logger.info("training completed.")


if __name__ == "__main__":
    feature_loader = FeatureLoader(file_name="feature_extracted")
    train_meta, train_embeddings = feature_loader.transform("train")

    pipeline = MapClustersIDsPipeline()
    pipeline.train_mapper(train_embeddings)
