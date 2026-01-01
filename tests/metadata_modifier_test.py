import numpy as np
import pandas as pd
from src.preprocessing.metadata_modifier import MetadataModifier


class TestMetadataModifier:
    def __init__(self):
        sample_meta_data = {
            "key_id": ["dev/12345_1_0", "dev/12345", "test/54321_2_1"],
            "feature1": [0.1, 0.2, 0.3],
            "feature2": [1.0, 1.1, 1.2],
        }
        self.sample_df = pd.DataFrame(sample_meta_data)
        self.current_config = "mls-tts-bark"
        self.audio_type = "spoof"
        self.speakers_ids = self.get_speakers_ids()
        self.modifier = MetadataModifier(audio_type=self.audio_type, speakers_ids=self.speakers_ids)

    def get_speakers_ids(self):
        np.random.seed(42)
        data = {
            "dev": pd.Series(np.random.randint(0, 100, size=67891)),
            "test": pd.Series(np.random.randint(0, 100, size=54322)),
        }
        return pd.DataFrame(data).fillna(-1).astype(int)

    def test_transform(self):
        modified_metadata = self.modifier.transform(self.current_config, self.sample_df)
        print("Modified Metadata DataFrame:")
        print(modified_metadata)

        expected_data = {
            "feature1": [0.1, 0.2, 0.3],
            "feature2": [1.0, 1.1, 1.2],
            "config": [self.current_config] * 3,
            "split": ["dev", "dev", "test"],
            "record_id": [12345, 12345, 54321],
            "speaker_id": [
                self.speakers_ids.loc[12345, "dev"],
                self.speakers_ids.loc[12345, "dev"],
                self.speakers_ids.loc[54321, "test"],
            ],
            "target": [self.audio_type] * 3,
        }
        expected_df = pd.DataFrame(expected_data)
        print("Expected Metadata DataFrame:")
        print(expected_df)

        pd.testing.assert_frame_equal(modified_metadata.reset_index(drop=True), expected_df.reset_index(drop=True))
        print("Successfully modified metadata:")


TestMetadataModifier().test_transform()
