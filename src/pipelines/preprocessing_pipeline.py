from src.preprocessing.config_loader import ConfigLoader
from src.preprocessing.audio_segmentator import AudioSegmentator
from src.preprocessing.wavlm_extractor import WavLmExtractor
from src.preprocessing.collector import Collector
from src.common.constants import Constants as consts
from src.common.constants import AudioType


class PreprocessingPipeline:
    def __init__(self, audio_type: AudioType):
        self.audio_type = audio_type
        print(audio_type)

    def serialize_waves_into_csv_with_embeddings(
        self, csv_file="extracted_embeddings.csv"
    ):
        config_loader = ConfigLoader(consts.tts_and_vocoders_configs)
        for data_set in config_loader.stream_next_config_dataset():
            print(
                f"Processing config: {config_loader.get_current_config()}, split: {config_loader.get_current_split()}"
            )
            segmented_audio_df, _ = AudioSegmentator().transform(data_set)
            print(
                f"Segmented audio into {len(segmented_audio_df)} segments."
            )
            embedding_audio_df = WavLmExtractor().transform(
                segmented_audio_df, consts.g_sample_rate
            )
            print(
                f"Extracted embeddings for {len(embedding_audio_df)} audio segments."
            )
            collector = Collector(save_file_name=csv_file)
            collector.transform(embedding_audio_df)
            break # To only check if it works

PreprocessingPipeline(consts.spoof).serialize_waves_into_csv_with_embeddings()