from datasets import load_dataset, Audio
from src.preprocessing.audio_segmentation import AudioSegmentator
from src.common.constants import Constants as const
class TestAudioSegmentation:
    def __init__(self):
        dataset = load_dataset("wqz995/AUDETER", "mls-tts-bark")
        dataset = dataset["dev"].cast_column("wav", Audio())
        self.dataset = dataset.select(range(2))
        
    def test_transform(self):
        audio_segmenter = AudioSegmentator()
        segmented_ds, durations_df= audio_segmenter.transform(self.dataset)

        assert segmented_ds.shape[0] > 2  # Each audio should be split into multiple segments
        
        print("Segmented Dataset Samples Shapes:")
        print(segmented_ds.iloc[0]["wave"].shape)
        print(segmented_ds.shape)
        print(segmented_ds.head(20))
        
        print("Durations DataFrame:")
        print(durations_df.head())
        
        


TestAudioSegmentation().test_transform()
