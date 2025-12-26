import numpy as np

from src.common.constants import Constants as consts
from src.preprocessing.audio_segmentator import AudioSegmentator
from src.preprocessing.collector import Collector
from src.preprocessing.config_loader import ConfigLoader
from src.preprocessing.wavlm_extractor import WavLmExtractor


class PreprocessingPipeline:
    def __init__(self, audio_type: str, config_lst: list[str] | None = None):
        self.audio_type = audio_type
        self.config_lst = config_lst

    def modify_audio_df(self, config_loader: ConfigLoader, segmented_audio_df):
        curr_cfg = config_loader.get_current_config()
        segmented_audio_df["key_id"] = segmented_audio_df["key_id"].apply(lambda x: f"{curr_cfg}_{x}")
        segmented_audio_df["target"] = self.audio_type
        return segmented_audio_df

    def serialize_waves_into_csv_with_embeddings(self, csv_file=consts.extracted_embeddings_csv, batch_size=8):
        config_loader = ConfigLoader(self.config_lst)
        for data_set in config_loader.stream_next_config_dataset():
            print(
                f"Processing config: {config_loader.get_current_config()}, split: {config_loader.get_current_split()}"
            )

            segmented_audio_df, _ = AudioSegmentator().transform(data_set)
            segmented_audio_df = self.modify_audio_df(config_loader, segmented_audio_df)
            print(f"Segmented audio into {len(segmented_audio_df)} segments.")

            embedding_audio_df = WavLmExtractor(batch_size=batch_size).transform(
                segmented_audio_df, consts.g_sample_rate
            )
            print(f"Extracted embeddings for {len(embedding_audio_df)} audio segments.")

            collector = Collector(save_file_name=csv_file)
            collector.transform(embedding_audio_df)


np.random.seed(42)
tts_sample = np.random.choice(consts.tts_configs, 2, replace=False)
vocoders_sample = np.random.choice(consts.vocoders_configs, 2, replace=False)
configs_lst = np.hstack([tts_sample, vocoders_sample]).tolist()
print("Selected configurations for preprocessing:", configs_lst)

BATCH_SIZE = 128

PreprocessingPipeline(consts.spoof, config_lst=configs_lst).serialize_waves_into_csv_with_embeddings(
    batch_size=BATCH_SIZE
)
