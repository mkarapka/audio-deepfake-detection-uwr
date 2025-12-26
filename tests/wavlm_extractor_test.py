import time

from datasets import Audio, load_dataset

from src.common.constants import Constants as consts
from src.common.logger import get_logger
from src.preprocessing.audio_segmentator import AudioSegmentator
from src.preprocessing.wavlm_extractor import WavLmExtractor

WAVLM_BASE_EMBEDDING_SIZE = 768


class TestWavLmExtractor:
    def __init__(self):
        self.logger = get_logger()
        dataset = load_dataset("wqz995/AUDETER", "mls-tts-bark", split="dev", streaming=True)
        self.dataset = dataset.cast_column("wav", Audio())

    def test_transform(self, batch_size=1, records_no=2):
        print(f"Testing WavLmExtractor with batch size: {batch_size}")
        dataset = [record for _, record in zip(range(records_no), self.dataset)]
        start = time.time()
        audio_segmenter = AudioSegmentator()
        segmented_df, _ = audio_segmenter.transform(dataset)

        wavlm_extractor = WavLmExtractor(batch_size=batch_size)
        features = wavlm_extractor.transform(segmented_df, consts.g_sample_rate)
        end = time.time()
        assert features.shape[0] == segmented_df.shape[0]
        assert all(len(emb) == WAVLM_BASE_EMBEDDING_SIZE for emb in features["wave"])

        self.logger.info(f"First embedding shape: {features['wave'].iloc[0].shape}")
        print(f"Time taken for batch size {batch_size}: {end - start:.4f} seconds")

    def test_transform_batch_size_one(self, records_no=2):
        self.test_transform(batch_size=1, records_no=records_no)

    def test_transform_batch_size_eight(self, records_no=100):
        self.test_transform(batch_size=8, records_no=records_no)

    def test_transform_batch_size_sixteen(self, records_no=100):
        self.test_transform(batch_size=16, records_no=records_no)

    def test_transform_batch_size_thirty_two(self, records_no=100):
        self.test_transform(batch_size=32, records_no=records_no)

    def test_transform_batch_size_sixty_four(self, records_no=1000):
        self.test_transform(batch_size=64, records_no=records_no)

    def test_transform_batch_size_one_twenty_eight(self, records_no=1000):
        self.test_transform(batch_size=128, records_no=records_no)

    def test_transform_batch_size_two_fifty_six(self, records_no=1000):
        self.test_transform(batch_size=256, records_no=records_no)


TestWavLmExtractor().test_transform_batch_size_sixty_four()
TestWavLmExtractor().test_transform_batch_size_one_twenty_eight()
TestWavLmExtractor().test_transform_batch_size_two_fifty_six()
