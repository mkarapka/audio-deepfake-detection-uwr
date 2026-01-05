import numpy as np
import pandas as pd

from src.common.constants import Constants as consts
from src.preprocessing.feature_loader import FeatureLoader

TEST_DIR = consts.collected_data_dir / "test_feature_loader"
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

    def test_transform(self):
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
        np.save(TEST_FILE_PATH.with_suffix(consts.embeddings_extension), sample_embeddings)
        for split in ["train", "dev", "test"]:
            sample_df.to_csv(
                f"{SPLIT_DIR}/{self.FILE_NAME}_{split}{consts.metadata_extension}",
                index=True,
            )

            # Wczytaj dane za pomocą FeatureLoader
            loaded_meta, loaded_embeddings = self.loader.transform(f"{split}")

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


TestFeatureLoader().test_transform()
