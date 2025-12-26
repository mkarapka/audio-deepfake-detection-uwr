import pandas as pd

from src.preprocessing.collector import Collector


class TestCollector:
    def test_transform(self):
        collector = Collector(save_file_name="test_collected_data.csv")

        # Create a sample DataFrame to collect
        sample_data = pd.DataFrame(
            {
                "key_id": ["id1", "id2", "id3"],
                "feature1": [0.1, 0.2, 0.3],
                "feature2": [1.0, 1.1, 1.2],
            }
        )

        # flush previous data
        if collector.save_file_path.exists():
            collector.save_file_path.unlink()

        # Transform (collect) the sample data
        collector.transform(sample_data)

        # Load the saved data to verify
        saved_data = pd.read_csv(collector.save_file_path)

        # Check if the saved data matches the original sample data
        pd.testing.assert_frame_equal(sample_data, saved_data)

        print("Collector transform test passed. Data collected and verified successfully.")

    def test_double_transform(self):
        collector = Collector(save_file_name="test_collected_data.csv")

        # Create a sample DataFrame to collect
        sample_data = pd.DataFrame({"key_id": ["id4", "id5"], "feature1": [0.4, 0.5], "feature2": [1.3, 1.4]})

        # flush previous data
        if collector.save_file_path.exists():
            collector.save_file_path.unlink()

        # Transform (collect) the sample data twice
        collector.transform(sample_data)
        collector.transform(sample_data)

        # Load the saved data to verify
        saved_data = pd.read_csv(collector.save_file_path)

        # Expected data is the original sample data repeated twice
        expected_data = pd.concat([sample_data, sample_data], ignore_index=True)
        print(expected_data)
        print(saved_data)
        # Check if the saved data matches the expected data
        pd.testing.assert_frame_equal(expected_data, saved_data)

        print("Collector double transform test passed. Data collected and verified successfully.")


TestCollector().test_transform()
TestCollector().test_double_transform()
