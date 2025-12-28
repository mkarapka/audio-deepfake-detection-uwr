import numpy as np

from src.common.constants import Constants as consts
from src.common.logger import get_logger, setup_logger
from src.preprocessing.audio_segmentator import AudioSegmentator
from src.preprocessing.collector import Collector
from src.preprocessing.config_loader import ConfigLoader
from src.preprocessing.wavlm_extractor import WavLmExtractor

LOGGER_NAME = "PreprocessingPipeline"
logger = get_logger(LOGGER_NAME)
setup_logger(LOGGER_NAME, log_to_console=True)


class PreprocessingPipeline:
    def __init__(self, audio_type: str, config_lst: list[str] | None = None):
        self.audio_type = audio_type
        self.config_lst = config_lst

    def _modify_audio_df(self, config_loader: ConfigLoader, segmented_audio_df):
        curr_cfg = config_loader.get_current_config()
        segmented_audio_df["key_id"] = segmented_audio_df["key_id"].apply(lambda x: f"{curr_cfg}_{x}")
        segmented_audio_df["target"] = self.audio_type
        return segmented_audio_df

    def save_audio_segments_embeddings_and_csv(self, file_name=consts.extracted_embeddings, batch_size=8):
        config_loader = ConfigLoader(self.config_lst)
        audio_segmentator = AudioSegmentator()
        wavlm_extractor = WavLmExtractor(batch_size=batch_size)
        collector = Collector(save_file_name=file_name)

        for data_set in config_loader.stream_next_config_dataset():
            logger.info(
                f"Processing config: {config_loader.get_current_config()}, split: {config_loader.get_current_split()}"
            )

            audio_segs_metadata, waves_segs = audio_segmentator.transform(data_set)
            audio_segs_metadata = self._modify_audio_df(config_loader, audio_segs_metadata)
            logger.info(f"Segmented audio into {audio_segs_metadata.shape[0]} segments.")

            embeddings = wavlm_extractor.transform(wave_segments=waves_segs, sample_rate=consts.g_sample_rate)
            logger.info(f"Extracted embeddings for {len(embeddings)} audio segments.")

            collector.transform(meta_df=audio_segs_metadata, embeddings=embeddings)
            logger.info(f"Saved embeddings and metadata for config: {config_loader.get_current_config()}")


np.random.seed(42)
tts_sample = np.random.choice(consts.tts_configs, 2, replace=False)
vocoders_sample = np.random.choice(consts.vocoders_configs, 2, replace=False)
configs_lst = np.hstack([tts_sample, vocoders_sample]).tolist()
logger.info(f"Selected configurations for preprocessing: {configs_lst}")

BATCH_SIZE = 8

PreprocessingPipeline(consts.spoof, config_lst=configs_lst).save_audio_segments_embeddings_and_csv(
    batch_size=BATCH_SIZE
)
