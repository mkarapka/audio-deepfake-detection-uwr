import numpy as np

from src.preprocessing.embedding_cluster_mapper import EmbeddingClusterMapper
from src.preprocessing.feature_loader import FeatureLoader


class MapClustersIDsPipeline:
    def train_mapper(self, train_embeddings: np.ndarray):
        mapper = EmbeddingClusterMapper(min_cluster_size=100, min_samples=50, random_state=42)
        mapper.logger.info("training cluster mapping model...")
        mapper.train(train_embeddings, n_components=10)
        mapper.logger.info("training completed.")


if __name__ == "__main__":
    feature_loader = FeatureLoader(file_name="feature_extracted")
    train_meta = feature_loader.load_split_file(split_name="train")
    reduced_train_meta = feature_loader.sample_fraction(train_meta, fraction=0.2)
    train_embeddings = feature_loader.load_embeddings_from_metadata(reduced_train_meta)

    pipeline = MapClustersIDsPipeline()
    pipeline.train_mapper(train_embeddings)
