import numpy as np

from src.common.basic_functions import measure_time
from src.preprocessing.audio_segmentator import AudioSegmentator
from src.preprocessing.feature_extractors.fft_extractor import FFTExtractor
from tests.base_test import BaseTest


class TestFFTExtractor(BaseTest):
    def __init__(self, records_no=500):
        super().__init__(records_no=records_no)
        print(f"Initialized TestFFTExtractor with {records_no} records.")

    @measure_time
    def test_transform(self, batch_size=2):
        print(f"Testing FFTExtractor with batch size: {batch_size}")
        audio_segmenter = AudioSegmentator()
        metadata, wave_segments = audio_segmenter.transform(self.dataset)
        print(f"Segmented audio into {wave_segments.shape} segments.")

        fft_extractor = FFTExtractor(batch_size=batch_size)
        features = fft_extractor.transform(wave_segments)

        assert features.shape[0] == wave_segments.shape[0] == metadata.shape[0]
        expected_feature_dim = (fft_extractor.window_size // 2 + 1) * 2
        assert np.all(features.shape[1] == expected_feature_dim)
        print(features.shape)
        print(features[0])

        self.logger.info(f"First feature vector shape: {features[0].shape}")
        print(f"FFTExtractor test passed for batch size {batch_size}.")

    def test_transform_batch_size_four(self):
        self.test_transform(batch_size=4)

    def test_transform_batch_size_eight(self):
        self.test_transform(batch_size=8)

    def test_transform_batch_size_sixteen(self):
        self.test_transform(batch_size=16)

    def test_transform_batch_size_thirty_two(self):
        self.test_transform(batch_size=32)

    def test_transform_batch_size_sixty_four(self):
        self.test_transform(batch_size=64)

    def test_transform_batch_size_one_twenty_eight(self):
        self.test_transform(batch_size=128)

    def test_transform_batch_size_two_fifty_six(self):
        self.test_transform(batch_size=256)


TestFFTExtractor().test_transform()
test2000 = TestFFTExtractor(records_no=2000)
test2000.test_transform_batch_size_eight()
test2000.test_transform_batch_size_sixteen()
test2000.test_transform_batch_size_thirty_two()
test2000.test_transform_batch_size_sixty_four()
