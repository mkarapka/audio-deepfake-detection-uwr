from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.collector import Collector

TEST_FILE_NAME = "test_file"


def delete_test_files(collector: Collector = None):
    meta_data_file_path = collector.get_metadata_file_path()
    embeddings_file_path = collector.get_embeddings_file_path()

    if meta_data_file_path.exists():
        meta_data_file_path.unlink()
    if embeddings_file_path.exists():
        embeddings_file_path.unlink()


def delete_test_splits_files(collector: Collector = None, splits=["train", "dev", "test"]):
    base_meta_path = collector.get_metadata_file_path()
    base_emb_path = collector.get_embeddings_file_path()

    for s_name in splits:
        meta_path = base_meta_path.parent / f"{base_meta_path.stem}_{s_name}{base_meta_path.suffix}"
        emb_path = base_emb_path.parent / f"{base_emb_path.stem}_{s_name}{base_emb_path.suffix}"

        if meta_path.exists():
            meta_path.unlink()
        if emb_path.exists():
            emb_path.unlink()


def change_file_path(collector: Collector):
    collector.data_dir = Path(str(collector.data_dir).replace("collected_data", "test_collected_data"))
    if not collector.data_dir.exists():
        collector.data_dir.mkdir(parents=True, exist_ok=True)
    collector.meta_data_file_path = collector.data_dir / Path(TEST_FILE_NAME + ".csv")
    collector.embeddings_file_path = collector.data_dir / Path(TEST_FILE_NAME + ".npy")


class TestCollector:
    def __init__(self):
        sample_meta_data = {
            "key_id": ["id1", "id2", "id3"],
            "feature1": [0.1, 0.2, 0.3],
            "feature2": [1.0, 1.1, 1.2],
        }
        self.sample_df = pd.DataFrame(sample_meta_data)

        self.embeddings_sample = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ]
        )

    def test_transform(self):
        collector = Collector(save_file_name=TEST_FILE_NAME)
        change_file_path(collector)
        delete_test_files(collector)

        # Transform (collect) the sample data
        collector.transform(meta_df=self.sample_df, embeddings=self.embeddings_sample)

        # Load the saved data to verify
        saved_metadata = pd.read_csv(collector.meta_data_file_path)
        saved_embeddings = np.load(collector.embeddings_file_path)

        # Check if the saved data matches the original sample data
        pd.testing.assert_frame_equal(self.sample_df, saved_metadata)
        np.testing.assert_array_equal(self.embeddings_sample, saved_embeddings)

        print("Collector transform test passed. Data collected and verified successfully.")

    def test_double_transform(self):
        collector = Collector(save_file_name=TEST_FILE_NAME)
        change_file_path(collector)
        delete_test_files(collector)

        sample_data = (self.sample_df, self.embeddings_sample)

        # Transform (collect) the sample data twice
        collector.transform(meta_df=sample_data[0], embeddings=sample_data[1])
        collector.transform(meta_df=sample_data[0], embeddings=sample_data[1])

        # Load the saved data to verify
        saved_metadata = pd.read_csv(collector.meta_data_file_path)
        saved_embeddings = np.load(collector.embeddings_file_path)

        # Expected data is the original sample data repeated twice
        expected_data = pd.concat([self.sample_df, self.sample_df], ignore_index=True)
        expected_embeddings = np.vstack([self.embeddings_sample, self.embeddings_sample])

        pd.testing.assert_frame_equal(expected_data, saved_metadata)
        np.testing.assert_array_equal(expected_embeddings, saved_embeddings)

        print("Collector double transform test passed. Data collected and verified successfully.")

    def test_transform_splits(self):
        collector = Collector(save_file_name=TEST_FILE_NAME)
        change_file_path(collector)
        delete_test_splits_files(collector)

        # Utwórz różne embeddingi dla każdego split
        emb1 = self.embeddings_sample
        emb2 = np.random.RandomState(42).rand(*self.embeddings_sample.shape)
        emb3 = np.random.RandomState(43).rand(*self.embeddings_sample.shape)

        embeddings = np.vstack([emb1, emb2, emb3])
        example_metadata = pd.concat([self.sample_df, self.sample_df, self.sample_df], ignore_index=True)

        first_part_meta = example_metadata.iloc[0:3]
        second_part_meta = example_metadata.iloc[3:6]
        third_part_meta = example_metadata.iloc[6:9]

        example_splits = [
            first_part_meta,
            second_part_meta,
            third_part_meta,
        ]

        collector.transform_splits(data=example_splits, splits=["train", "dev", "test"])

        # Load saved splits to verify
        base_meta_path = collector.get_metadata_file_path()

        # Train split
        meta_path_train = base_meta_path.parent / f"{base_meta_path.stem}_train{base_meta_path.suffix}"
        saved_metadata_train = pd.read_csv(meta_path_train, index_col=0)
        emb_train = embeddings[saved_metadata_train.index]

        # Dev split
        meta_path_dev = base_meta_path.parent / f"{base_meta_path.stem}_dev{base_meta_path.suffix}"
        saved_metadata_dev = pd.read_csv(meta_path_dev, index_col=0)
        emb_dev = embeddings[saved_metadata_dev.index]

        # Test split
        meta_path_test = base_meta_path.parent / f"{base_meta_path.stem}_test{base_meta_path.suffix}"
        saved_metadata_test = pd.read_csv(meta_path_test, index_col=0)
        emb_test = embeddings[saved_metadata_test.index]

        # Sprawdź czy zapisane dane są poprawne
        print(self.sample_df)
        print("---")
        print(saved_metadata_train)
        print("---")
        print(saved_metadata_dev)
        print("---")
        print(saved_metadata_test)
        pd.testing.assert_frame_equal(first_part_meta, saved_metadata_train), "Train metadata does not match."
        np.testing.assert_array_equal(emb1, emb_train), "Train embeddings do not match."
        pd.testing.assert_frame_equal(second_part_meta, saved_metadata_dev), "Dev metadata does not match."
        np.testing.assert_array_equal(emb2, emb_dev), "Dev embeddings do not match."
        pd.testing.assert_frame_equal(third_part_meta, saved_metadata_test), "Test metadata does not match."
        np.testing.assert_array_equal(emb3, emb_test), "Test embeddings do not match."
        print("Collector transform_splits test passed. All splits saved and verified successfully.")


TestCollector().test_transform()
TestCollector().test_double_transform()
TestCollector().test_transform_splits()
