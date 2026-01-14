import numpy as np
import pandas as pd

from src.preprocessing.collector import Collector
from src.preprocessing.embedding_cluster_mapper import EmbeddingClusterMapper
from src.preprocessing.feature_loader import FeatureLoader


class MapClustersIDsPipeline:
    def plot_cluster_info(self, train_embeddings: np.ndarray):
        mapper = EmbeddingClusterMapper(K_list=[2, 4, 8, 10, 16, 32], random_state=42)
        silhouette_scores, inertias = mapper.get_info_for_each_cluster(train_embeddings)

        print("Silhouette Scores:", silhouette_scores)
        print("Inertias:", inertias)

        collector = Collector(save_file_name="cluster_analysis")
        silhouette_df = pd.DataFrame(list(silhouette_scores.items()), columns=["n_clusters", "silhouette_score"])
        inertia_df = pd.DataFrame(list(inertias.items()), columns=["n_clusters", "inertia"])
        collector._write_data_to_csv(silhouette_df, file_path=collector._create_file_path("_silhouette.csv"))
        collector._write_data_to_csv(inertia_df, file_path=collector._create_file_path("_inertia.csv"))


if __name__ == "__main__":
    feature_loader = FeatureLoader(file_name="feature_extracted")
    train_meta, train_embeddings = feature_loader.transform("train")

    pipeline = MapClustersIDsPipeline()
    pipeline.plot_cluster_info(train_embeddings)
