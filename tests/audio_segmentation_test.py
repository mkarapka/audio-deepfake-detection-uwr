import pandas as pd
from src.preprocessing.audio_segmentator import AudioSegmentator

from tests.base_test import BaseTest


class TestAudioSegmentation(BaseTest):
    def __init__(self):
        super().__init__(split="dev", config="mls-tts-bark", records_no=2)

    def calculate_number_of_segments(self, duration, chunk_sec, overlap_sec):
        stride = chunk_sec - overlap_sec
        if duration <= chunk_sec:
            return 1
        else:
            num_segments = 1 + int((duration - chunk_sec) / stride)
            if (duration - chunk_sec) % stride > 0:
                num_segments += 1
            return num_segments

    def get_durrations_from_dataset(self, dataset):
        durations = []
        for record in dataset:
            samples = record["wav"]["array"]
            sr = record["wav"]["sampling_rate"]
            dur = samples.shape[0] / sr
            durations.append({"key_id": record["__key__"], "duration": dur})
        return pd.DataFrame(durations)

    def test_transform(self):
        audio_segmentator = AudioSegmentator()
        metadata, wave_segments = audio_segmentator.transform(self.dataset)
        durations_df = self.get_durrations_from_dataset(self.dataset)

        # Each audio should be split into multiple segments
        assert metadata.shape[0] >= 2 and wave_segments.shape[0] >= 2
        assert metadata.shape[0] == wave_segments.shape[0]

        assert self.calculate_number_of_segments(
            duration=durations_df["duration"].iloc[0],
            chunk_sec=audio_segmentator.chunk_sec,
            overlap_sec=audio_segmentator.overlap_sec,
        ) == len(metadata[metadata["key_id"] == durations_df["key_id"].iloc[0]])

        assert self.calculate_number_of_segments(
            duration=durations_df["duration"].iloc[1],
            chunk_sec=audio_segmentator.chunk_sec,
            overlap_sec=audio_segmentator.overlap_sec,
        ) == len(metadata[metadata["key_id"] == durations_df["key_id"].iloc[1]])

        print("Segmented Dataset Samples Shapes:")
        print("Wave Segments Shape:", wave_segments.shape)
        print("Wave Segment[0] Shape:", wave_segments[0].shape)

        print("Metadata Shape:", metadata.shape)
        print(metadata.head(20))

        print("Durations DataFrame:")
        print(durations_df.head())

    def test_transform_shorter_than_4_seconds(self):
        self.dataset[0]["wav"]["array"] = self.dataset[0]["wav"]["array"][:8000]  # 0.5 seconds at 16kHz
        audio_segmentator = AudioSegmentator()
        metadata, wave_segments = audio_segmentator.transform(self.dataset)

        assert metadata.shape[0] >= 2  # First audio should yield 1 segment, second as before
        assert wave_segments[0].shape[0] == 64000  # Padded to 4 seconds (64000 samples at 16kHz)
        print("Tested segmentation for audio shorter than 4 seconds successfully.")


TestAudioSegmentation().test_transform()
TestAudioSegmentation().test_transform_shorter_than_4_seconds()
