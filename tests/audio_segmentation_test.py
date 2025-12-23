from datasets import load_dataset, Audio
from src.preprocessing.audio_segmentator import AudioSegmentator
class TestAudioSegmentation:
    def __init__(self):
        dataset = load_dataset("wqz995/AUDETER", "mls-tts-bark", split="dev", streaming=True)
        dataset = dataset.cast_column("wav", Audio())
        self.dataset = [record for _, record in zip(range(2), dataset)]
        
    def calculate_number_of_segments(self, duration, chunk_sec, overlap_sec):
        stride = chunk_sec - overlap_sec
        if duration <= chunk_sec:
            return 1
        else:
            num_segments = 1 + int((duration - chunk_sec) / stride)
            if (duration - chunk_sec) % stride > 0:
                num_segments += 1
            return num_segments
        
    def test_transform(self):
        audio_segmenter = AudioSegmentator()
        segmented_ds, durations_df = audio_segmenter.transform(self.dataset)

        assert segmented_ds.shape[0] > 2  # Each audio should be split into multiple segments
        assert self.calculate_number_of_segments(
            duration=durations_df['duration'].iloc[0],
            chunk_sec=audio_segmenter.chunk_sec,
            overlap_sec=audio_segmenter.overlap_sec
        ) == len(segmented_ds[segmented_ds['key_id'] == durations_df['key_id'].iloc[0]])
        
        assert self.calculate_number_of_segments(
            duration=durations_df['duration'].iloc[1],
            chunk_sec=audio_segmenter.chunk_sec,
            overlap_sec=audio_segmenter.overlap_sec
        ) == len(segmented_ds[segmented_ds['key_id'] == durations_df['key_id'].iloc[1]])
        
        print("Segmented Dataset Samples Shapes:")
        print(segmented_ds.iloc[0]["wave"].shape)
        print(segmented_ds.shape)
        print(segmented_ds.head(20))
        
        print("Durations DataFrame:")
        print(durations_df.head())
        
        


TestAudioSegmentation().test_transform()
