import pandas as pd
import numpy as np
from src.preprocessing.collector import Collector

TEST_FILE_NAME = "test_collected_data"

def delete_test_files(collector : Collector = None):
    meta_data_file_path = collector.get_meta_data_file_path()
    embeddings_file_path = collector.get_embeddings_file_path()
    
    if meta_data_file_path.exists():
        meta_data_file_path.unlink()
    if embeddings_file_path.exists():
        embeddings_file_path.unlink()

class TestCollector:
    def __init__(self):
        sample_meta_data = {
            "key_id": ["id1", "id2", "id3"],
            "feature1": [0.1, 0.2, 0.3],
            "feature2": [1.0, 1.1, 1.2],
        }
        self.sample_df = pd.DataFrame(sample_meta_data)
        
        self.embeddings_sample = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ])
    
    def test_transform(self):
        collector = Collector(save_file_name=TEST_FILE_NAME)
        delete_test_files(collector)

        # Transform (collect) the sample data
        collector.transform((self.sample_df, self.embeddings_sample))

        # Load the saved data to verify
        saved_metadata = pd.read_csv(collector.meta_data_file_path)
        saved_embeddings = np.load(collector.embeddings_file_path)

        # Check if the saved data matches the original sample data
        pd.testing.assert_frame_equal(self.sample_df, saved_metadata)
        np.testing.assert_array_equal(self.embeddings_sample, saved_embeddings)

        print("Collector transform test passed. Data collected and verified successfully.")

    def test_double_transform(self):
        collector = Collector(save_file_name=TEST_FILE_NAME)
        delete_test_files(collector)

        sample_data = (self.sample_df, self.embeddings_sample)
        
        # Transform (collect) the sample data twice
        collector.transform(sample_data)
        collector.transform(sample_data)

        # Load the saved data to verify
        saved_metadata = pd.read_csv(collector.meta_data_file_path)
        saved_embeddings = np.load(collector.embeddings_file_path)

        # Expected data is the original sample data repeated twice
        expected_data = pd.concat([self.sample_df, self.sample_df], ignore_index=True)
        expected_embeddings = np.vstack([self.embeddings_sample, self.embeddings_sample])
        
        pd.testing.assert_frame_equal(expected_data, saved_metadata)
        np.testing.assert_array_equal(expected_embeddings, saved_embeddings)

        print("Collector double transform test passed. Data collected and verified successfully.")


TestCollector().test_transform()
TestCollector().test_double_transform()
