import time

import numpy as np

from src.preprocessing.audio_segmentator import AudioSegmentator
from src.preprocessing.feature_extractors.wavlm_extractor import WavLmExtractor
from tests.base_test import BaseTest

WAVLM_BASE_EMBEDDING_SIZE = 768


class TestWavLmExtractor(BaseTest):
    def __init__(self, records_no=2):
        super().__init__(records_no=records_no)

    def test_transform(self, batch_size=1):
        print(f"Testing WavLmExtractor with batch size: {batch_size}")
        start = time.time()
        audio_segmenter = AudioSegmentator()
        metadata, wave_segments = audio_segmenter.transform(self.dataset)

        wavlm_extractor = WavLmExtractor(batch_size=batch_size)
        features = wavlm_extractor.transform(wave_segments)
        end = time.time()

        assert features.shape[0] == wave_segments.shape[0] == metadata.shape[0]
        assert np.all(features.shape[1] == WAVLM_BASE_EMBEDDING_SIZE)

        self.logger.info(f"First embedding shape: {features[0].shape}")
        print(f"Time taken for batch size {batch_size}: {end - start:.4f} seconds")

    def test_transform_batch_size_one(self):
        self.test_transform(batch_size=1)

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


TestWavLmExtractor().test_transform_batch_size_one()
TestWavLmExtractor(records_no=10).test_transform_batch_size_eight()

# test_wavlm_1000 = TestWavLmExtractor(records_no=1000)
# test_wavlm_1000.test_transform_batch_size_sixteen()
# test_wavlm_1000.test_transform_batch_size_thirty_two()
# test_wavlm_1000.test_transform_batch_size_sixty_four()
# test_wavlm_1000.test_transform_batch_size_one_twenty_eight()
# test_wavlm_1000.test_transform_batch_size_two_fifty_six()
