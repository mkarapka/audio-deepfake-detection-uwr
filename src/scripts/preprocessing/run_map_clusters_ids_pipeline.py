from src.common.constants import Constants as consts
from src.pipelines.map_clusters_ids_pipeline import MapClustersIDsPipeline

if __name__ == "__main__":
    output_file = "mapped_clusters_ids_feature_extracted"
    umap_config = consts.umap_20d_config
    hdbscan_config = consts.hdbscan_config

    pipeline = MapClustersIDsPipeline(
        output_file=output_file,
        umap_config=umap_config,
        hdbscan_config=hdbscan_config,
    )
    pipeline.map_clusters(fraction=0.1)
