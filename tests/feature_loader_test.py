import numpy as np
import pandas as pd

from src.common.constants import Constants as consts
from src.preprocessing.io.feature_loader import FeatureLoader

TEST_DIR = consts.tests_data_dir / "test_feature_loader"
TEST_FILE_PATH = TEST_DIR / "test_file"
TEST_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_DIR = TEST_DIR / "splited_data"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)


def delete_test_files():
    if SPLIT_DIR.exists() is False:
        SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    files = SPLIT_DIR.iterdir()
    for file in files:
        file.unlink()


class TestFeatureLoader:
    def __init__(self):
        self.FILE_NAME = "test_file"
        self.loader = FeatureLoader(
            file_name=self.FILE_NAME,
            data_dir=TEST_DIR,
            split_dir=SPLIT_DIR,
        )

    def test_load_data_split(self):
        delete_test_files()

        # Przygotuj przykładowe dane do zapisania
        sample_meta_data = {
            "key_id": ["id1", "id2", "id3"],
            "feature1": [0.1, 0.2, 0.3],
            "feature2": [1.0, 1.1, 1.2],
        }
        sample_df = pd.DataFrame(sample_meta_data)
        sample_embeddings = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ]
        )
        np.save(TEST_FILE_PATH.with_suffix(consts.npy_ext), sample_embeddings)
        for split in ["train", "dev", "test"]:
            sample_df.to_csv(
                f"{SPLIT_DIR}/{self.FILE_NAME}_{split}{consts.csv_ext}",
                index=True,
            )

            # Wczytaj dane za pomocą FeatureLoader
            loaded_meta, loaded_embeddings = self.loader.load_data_split(f"{split}")

            # Sprawdź czy wczytane dane są poprawne
            (
                pd.testing.assert_frame_equal(sample_df, loaded_meta),
                "Metadata does not match.",
            )
            (
                np.testing.assert_array_equal(sample_embeddings, loaded_embeddings),
                "Embeddings do not match.",
            )

            print("FeatureLoader transform test passed. Metadata and embeddings match successfully.")

    def test_load_speakers_ids(self):
        delete_test_files()

        # Przygotuj przykładowe dane do zapisania
        sample_speakers_data = {
            "speaker_id": ["spk1", "spk2", "spk3"],
            "attribute": ["attr1", "attr2", "attr3"],
        }
        sample_speakers_df = pd.DataFrame(sample_speakers_data)
        speakers_ids_path = TEST_DIR / consts.speakers_ids_file
        sample_speakers_df.to_csv(speakers_ids_path, index=False)

        # Wczytaj dane za pomocą FeatureLoader
        loaded_speakers_df = self.loader.load_speakers_ids()

        # Sprawdź czy wczytane dane są poprawne
        (
            pd.testing.assert_frame_equal(sample_speakers_df, loaded_speakers_df),
            "Speakers IDs data does not match.",
        )

        print("FeatureLoader load_speakers_ids test passed. Speakers IDs data match successfully.")

    def test_load_metadata_file_when_there_is_no_index_in_column_zero(self):
        delete_test_files()

        # Przygotuj przykładowe dane do zapisania
        sample_meta_data = {
            "key_id": ["id1", "id2", "id3"],
            "feature1": [0.1, 0.2, 0.3],
            "feature2": [1.0, 1.1, 1.2],
        }
        sample_df = pd.DataFrame(sample_meta_data)
        sample_df.to_csv(
            TEST_DIR / f"{self.FILE_NAME}_metadata{consts.csv_ext}",
            index=False,
        )

        # Wczytaj dane za pomocą FeatureLoader
        loaded_meta = self.loader.load_metadata_file(TEST_DIR / f"{self.FILE_NAME}_metadata{consts.csv_ext}")

        print("Loaded metadata index:")
        print(loaded_meta.index)
        print("Saved metadata index:")
        print(sample_df.index)

        # Sprawdź czy wczytane dane są poprawne
        (
            pd.testing.assert_frame_equal(sample_df, loaded_meta),
            "Metadata does not match.",
        )

        print("FeatureLoader load_metadata_file test passed. Metadata match successfully.")

    def test_load_data(self):
        delete_test_files()

        # Przygotuj przykładowe dane do zapisania
        sample_meta_data = {
            "key_id": ["id1", "id2", "id3"],
            "feature1": [0.1, 0.2, 0.3],
            "feature2": [1.0, 1.1, 1.2],
        }
        sample_df = pd.DataFrame(sample_meta_data)
        sample_embeddings = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ]
        )
        np.save(TEST_FILE_PATH.with_suffix(consts.npy_ext), sample_embeddings)
        sample_df.to_csv(
            TEST_DIR / f"{self.FILE_NAME}{consts.csv_ext}",
            index=False,
        )

        # Wczytaj dane za pomocą FeatureLoader
        loaded_meta, loaded_embeddings = self.loader.load_data()
        # Sprawdź czy wczytane dane są poprawne
        (
            pd.testing.assert_frame_equal(sample_df, loaded_meta),
            "Metadata does not match.",
        )
        (
            np.testing.assert_array_equal(sample_embeddings, loaded_embeddings),
            "Embeddings do not match.",
        )
        _, counts = np.unique(loaded_meta.index, return_counts=True)
        assert all(count == 1 for count in counts), "There are duplicate indices in loaded metadata."
        print(loaded_meta)
        print("FeatureLoader load_data test passed. Metadata and embeddings match successfully.")

    def test_sample_data(self):
        simple_metadata = {
            "key_id": np.random.randint(0, 100, size=100),
            "value": np.random.rand(100),
        }
        simple_df = pd.DataFrame(simple_metadata)
        sampled_df = self.loader.sample_data(metadata=simple_df, fraction=0.1)
        print(sampled_df.shape)
        assert len(sampled_df) == 10, f"Expected 10 samples, got {len(sampled_df)}"

    def test_sample_data_with_embeddings(self):
        simple_metadata = {
            "key_id": np.random.randint(0, 100, size=100),
            "value": np.random.rand(100),
        }
        simple_df = pd.DataFrame(simple_metadata)
        simple_embeddings = np.random.rand(100, 5)
        sampled_df, sampled_embeddings = self.loader.sample_data(
            metadata=simple_df, embeddings=simple_embeddings, fraction=0.1
        )
        print(sampled_df.shape, sampled_embeddings.shape)
        assert len(sampled_df) == 10, f"Expected 10 samples, got {len(sampled_df)}"
        assert sampled_embeddings.shape == (10, 5), f"Expected embeddings shape (10, 5), got {sampled_embeddings.shape}"

    def test_sample_data_by_audio_id(self):
        simple_metadata = {
            "audio_id": np.random.choice(np.arange(0, 10, 1), size=100),
            "value": np.random.rand(100),
        }
        simple_df = pd.DataFrame(simple_metadata)
        sampled_df = self.loader.sample_data_by_audio_id(metadata=simple_df, fraction=0.4)
        print(type(sampled_df))
        print(sampled_df)
        print(sampled_df["audio_id"].unique())
        assert len(sampled_df["audio_id"].unique()) == 4
        print("Sampled_df shape:")
        print(sampled_df.shape)

    def test_sample_data_by_audio_id_with_embeddings(self):
        simple_metadata = {
            "audio_id": np.random.choice(np.arange(0, 10, 1), size=100),
            "value": np.random.rand(100),
            "other_value": np.random.rand(100),
        }
        simple_emb = np.random.rand(100, 5)
        simple_df = pd.DataFrame(simple_metadata)
        sampled_df, sampled_emb = self.loader.sample_data_by_audio_id(
            metadata=simple_df, embeddings=simple_emb, fraction=0.4
        )

        assert len(sampled_df["audio_id"].unique()) == 4
        print("Sampled_df shapes:")
        print(sampled_df.shape, sampled_emb.shape)


TestFeatureLoader().test_load_data_split()
TestFeatureLoader().test_load_speakers_ids()
TestFeatureLoader().test_load_metadata_file_when_there_is_no_index_in_column_zero()
TestFeatureLoader().test_load_data()
TestFeatureLoader().test_sample_data()
TestFeatureLoader().test_sample_data_with_embeddings()
TestFeatureLoader().test_sample_data_by_audio_id()
TestFeatureLoader().test_sample_data_by_audio_id_with_embeddings()
