import pandas as pd

from src.preprocessing.unique_audio_id_mapper import UniqueAudioIdMapper


class TestUniqueAudioIDMapper:
    def test_transform(self):
        data = {
            "config": ["cfg1", "cfg1", "cfg2", "cfg2", "cfg1"],
            "split": ["train", "train", "test", "test", "train"],
            "record_id": [1, 2, 1, 2, 1],
        }
        df = pd.DataFrame(data)
        df_cp = df.copy()
        mapper = UniqueAudioIdMapper()
        transformed_df = mapper.transform(df_cp)

        expected_ids = [0, 1, 2, 3, 0]
        assert transformed_df["unique_audio_id"].tolist() == expected_ids
        assert transformed_df.shape[0] == df.shape[0]
        assert "unique_audio_id" in transformed_df.columns
        pd.testing.assert_frame_equal(transformed_df.drop(columns=["unique_audio_id"]), df)
        print("test_transform passed")


TestUniqueAudioIDMapper().test_transform()
