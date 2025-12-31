from src.preprocessing.metadata_modifier import MetadataModifier
import pandas as pd

class TestMetadataModifier:
    def __init__(self):
        sample_meta_data = {
            "key_id": ["dev/12345_1_0", "dev/67890", "test/54321_2_1"],
            "feature1": [0.1, 0.2, 0.3],
            "feature2": [1.0, 1.1, 1.2],
        }
        self.sample_df = pd.DataFrame(sample_meta_data)
        self.current_config = "mls-tts-bark"
        self.audio_type = "spoof"
        self.modifier = MetadataModifier(audio_type=self.audio_type)

    def test_transform(self):
        modified_metadata = self.modifier.transform(self.current_config, self.sample_df)
        print("Modified Metadata DataFrame:")
        print(modified_metadata)
        
        expected_data = {
            "feature1": [0.1, 0.2, 0.3],
            "feature2": [1.0, 1.1, 1.2],
            "config": [self.current_config] * 3,
            "split": ["dev", "dev", "test"],
            "record_id": [12345, 67890, 54321],
            "target": [self.audio_type] * 3,
        }
        expected_df = pd.DataFrame(expected_data)
        print("Expected Metadata DataFrame:")
        print(expected_df)

        pd.testing.assert_frame_equal(modified_metadata.reset_index(drop=True), expected_df.reset_index(drop=True))
        print("Successfully modified metadata:")
        
TestMetadataModifier().test_transform()