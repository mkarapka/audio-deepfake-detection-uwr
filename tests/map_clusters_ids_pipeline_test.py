from pathlib import Path

import numpy as np
import pandas as pd

from src.common.constants import Constants as consts
from src.common.logger import setup_logger
from src.pipelines.map_clusters_ids_pipeline import MapClustersIDsPipeline
from src.preprocessing.io.collector import Collector
from src.preprocessing.io.feature_loader import FeatureLoader

TEST_DATA_SIZE = 10000


class TestMapClustersIDsPipeline:
    def __init__(self):
        self.logger = setup_logger(__class__.__name__, log_to_console=True)
        test_data_dir_path = consts.data_dir / "tests_data"
        tests_data_splited_dir_path = test_data_dir_path / "splited_data"
        if not test_data_dir_path.exists():
            test_data_dir_path.mkdir(parents=True, exist_ok=True)
        if not tests_data_splited_dir_path.exists():
            tests_data_splited_dir_path.mkdir(parents=True, exist_ok=True)

        test_all_meta_file_path = consts.data_dir / "tests_data/test_feature_extracted.csv"
        test_all_emb_file_path = consts.data_dir / "tests_data/test_feature_extracted.npy"
        test_train_meta_file_path = consts.data_dir / "tests_data/splited_data/test_feature_extracted_train.csv"
        test_train_emb_file_path = consts.data_dir / "tests_data/splited_data/test_feature_extracted_train.npy"

        if test_all_meta_file_path.exists():
            test_all_meta_file_path.unlink()
        if test_all_emb_file_path.exists():
            test_all_emb_file_path.unlink()
        if test_train_meta_file_path.exists():
            test_train_meta_file_path.unlink()
        if test_train_emb_file_path.exists():
            test_train_emb_file_path.unlink()

        tmp_feature_loader = FeatureLoader(consts.feature_extracted)
        tmp_all_meta, tmp_all_embeddings = tmp_feature_loader.load_data_split(split_name="train")
        test_all_metadata = tmp_all_meta.iloc[:TEST_DATA_SIZE]
        test_all_embeddings = tmp_all_embeddings[:TEST_DATA_SIZE]

        test_train_metadata = test_all_metadata.iloc[: TEST_DATA_SIZE // 2].reset_index(drop=True)
        tests_train_embeddings = test_all_embeddings[: TEST_DATA_SIZE // 2]

        test_all_metadata.to_csv(test_all_meta_file_path, index=False)
        np.save(test_all_emb_file_path, test_all_embeddings)
        test_train_metadata.to_csv(test_train_meta_file_path, index=True)
        np.save(test_train_emb_file_path, tests_train_embeddings)

        print("Loaded test data for MapClustersIDsPipeline tests.")
        print("Loaded all metadata:", pd.read_csv(test_all_meta_file_path))
        print("All metadata and embeddings shapes:", test_all_metadata.shape, test_all_embeddings.shape)
        print("Loaded train metadata:", pd.read_csv(test_train_meta_file_path))
        print("Train metadata and embeddings shapes:", test_train_metadata.shape, tests_train_embeddings.shape)

    def test_map_clusters_ids_pipeline(self):
        umap_config = consts.umap_20d_config
        hdbscan_config = consts.hdbscan_config

        pipeline = MapClustersIDsPipeline(
            output_file="test_output",
            umap_config=umap_config,
            hdbscan_config=hdbscan_config,
        )

        pipeline.feature_loader = FeatureLoader(
            file_name="test_feature_extracted",
            data_dir=Path(consts.data_dir / "tests_data"),
            split_dir=Path(consts.data_dir / "tests_data/splited_data"),
        )
        pipeline.collector = Collector(
            save_file_name=pipeline.output_file,
            data_dir=Path(consts.data_dir / "tests_data"),
            split_dir=Path(consts.data_dir / "tests_data/splited_data"),
        )
        if Path(pipeline.collector.get_metadata_file_path()).exists():
            Path(pipeline.collector.get_metadata_file_path()).unlink()
        pipeline.map_clusters(fraction=0.5)

        assert pipeline.output_file == "test_output"
        print(pipeline.collector.meta_data_file_path, "dupa")
        test_output = pd.read_csv(pipeline.collector.meta_data_file_path)
        print("Test output loaded:", test_output)
        assert test_output["cluster_id"].notna().any()
        assert test_output["cluster_strength"].notna().any()
        assert test_output.shape[0] == TEST_DATA_SIZE
        print(test_output.head())
        print("test_map_clusters_ids_pipeline passed.")


TestMapClustersIDsPipeline().test_map_clusters_ids_pipeline()
