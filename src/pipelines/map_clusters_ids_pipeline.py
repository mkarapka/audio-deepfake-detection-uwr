import numpy as np
from hdbscan import HDBSCAN, approximate_predict
from umap import UMAP

from src.common.basic_functions import setup_logger
from src.common.constants import Constants as consts
from src.preprocessing.collector import Collector
from src.preprocessing.feature_loader import FeatureLoader


class MapClustersIDsPipeline:
    def __init__(
        self,
        output_file: str,
        umap_config: dict[str, any],
        hdbscan_config: dict[str, any],
    ):
        self.logger = setup_logger(__class__.__name__, log_to_console=True)
        self.output_file = output_file

        self.feature_loader = FeatureLoader(file_name=consts.feature_extracted)
        self.collector = Collector(save_file_name=self.output_file)

        self.umap_model = UMAP(**umap_config)
        self.hdbscan_model = HDBSCAN(**hdbscan_config)

    def _log_summary_of_clusters(self, clusters: np.ndarray):
        unique, counts = np.unique(clusters, return_counts=True)
        cluster_summary = dict(zip(unique.tolist(), counts.tolist()))
        self.logger.info(f"Cluster summary: {cluster_summary}")

    def map_clusters(self, fraction: float = 0.1):
        self.logger.info("Loading and sampling training data for UMAP and HDBSCAN fitting.")
        train_meta, train_embeddings = self.feature_loader.load_data_split(split_name="train")
        sampled_train_meta = self.feature_loader.sample_data(train_meta, fraction=fraction)
        sampled_train_embeddings = train_embeddings[sampled_train_meta.index]
        sampled_train_meta = sampled_train_meta.reset_index(drop=True)

        self.logger.info("Start UMAP transformation.")
        sampled_X_train_emb_umap = self.umap_model.fit_transform(sampled_train_embeddings)

        self.logger.info("Start HDBSCAN clustering.")
        self.hdbscan_model.fit(sampled_X_train_emb_umap)

        self.logger.info("Start loading all data.")
        all_metadata = self.feature_loader.load_metadata_file()
        all_embeddings = self.feature_loader.load_embeddings_from_metadata(all_metadata)

        self.logger.info("Start mapping clusters to all data.")
        all_embeddings_umap = self.umap_model.transform(all_embeddings)
        clusters, strengths = approximate_predict(self.hdbscan_model, all_embeddings_umap)
        all_metadata["cluster_id"] = clusters
        all_metadata["cluster_strength"] = strengths
        self._log_summary_of_clusters(clusters)

        self.logger.info("Mapping clusters completed, saving data.")
        self.collector.write_data_to_csv(data=all_metadata)
        self.logger.info("Clusters mapping completed and data saved.")
