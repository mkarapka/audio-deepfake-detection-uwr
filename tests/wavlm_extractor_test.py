from datasets import Audio, load_dataset

from preprocessing.audio_segmentator import AudioSegmentator
from src.common.constants import Constants as const
from src.preprocessing.wavlm_extractor import WavLmExtractor


class TestWavLmExtractor:
    def __init__(self):
        dataset = load_dataset("wqz995/AUDETER", "mls-tts-bark")
        dataset = dataset["dev"].cast_column("wav", Audio())
        self.dataset = dataset.select(range(2))

    def test_transform(self):
        audio_segmenter = AudioSegmentator()
        segmented_df, _ = audio_segmenter.transform(self.dataset)

        wavlm_extractor = WavLmExtractor(pretrained_model_name=const.wavlm_base_plus_name)
        features = wavlm_extractor.transform(segmented_df, const.g_sample_rate)

        print(features.shape)
        print(features)
        assert features.shape[0] == segmented_df.shape[0]
        assert features["wave"].iloc[0].shape[0] == 768  # Assuming WavLM Base model output dimension
        print(features["wave"].iloc[0].shape)


TestWavLmExtractor().test_transform()
