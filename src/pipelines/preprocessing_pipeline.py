from src.common.constants import Constants as consts
from src.common.logger import get_logger, setup_logger
from src.preprocessing.audio_segmentator import AudioSegmentator
from src.preprocessing.collector import Collector
from src.preprocessing.config_loader import ConfigLoader
from src.preprocessing.feature_extractors.base_feature_extractor import (
    BaseFeatureExtractor,
)
from src.preprocessing.feature_extractors.fft_extractor import FFTExtractor
from src.preprocessing.feature_extractors.wavlm_extractor import WavLmExtractor
from src.preprocessing.metadata_modifier import MetadataModifier

LOGGER_NAME = "PreprocessingPipeline"
logger = get_logger(LOGGER_NAME)
setup_logger(LOGGER_NAME, log_to_console=True)


class PreprocessingPipeline:
    def __init__(self, audio_type: str, config_lst: list[str] | None = None):
        self.audio_type = audio_type
        self.config_lst = config_lst
        self.source_dataset = consts.audeter_ds_path if audio_type == consts.spoof else consts.mls_eng_ds_path

        logger.info(f"Initialized PreprocessingPipeline for audio type: {self.audio_type}")
        logger.info(f"Source dataset set to: {self.source_dataset}")
        logger.info(f"Configurations to be used: {self.config_lst}")
        if self.audio_type not in [consts.spoof, consts.bonafide]:
            logger.error(f"Invalid audio type: {self.audio_type}. Must be 'spoof' or 'bonafide'.")

    def _normalize_dataset_format(self, ds, split):
        for idx, record in enumerate(ds):
            if "__key__" not in record:
                record["__key__"] = f"{split}/{idx}_0"
            if "audio" in record and "wav" not in record:
                record["wav"] = record.pop("audio")
            yield record

    def _preprocess_dataset(self, file_name: str, feature_extractor: BaseFeatureExtractor, batch_size=8):
        if file_name is None or file_name == "":
            logger.error("File name for saving processed data must be provided.")
        if feature_extractor is None:
            logger.error("Feature extractor must be provided for preprocessing.")

        config_loader = ConfigLoader(source_dataset=self.source_dataset, config=self.config_lst)
        audio_segmentator = AudioSegmentator()
        metadata_modifier = MetadataModifier(audio_type=self.audio_type, speakers_ids=config_loader.load_speakers_ids())
        collector = Collector(save_file_name=file_name)

        for dataset_split in config_loader.stream_next_config_dataset():
            logger.info(
                f"Processing data set {self.source_dataset}, config: {
                    config_loader.get_current_config()
                }, split: {config_loader.get_current_split()}"
            )
            if self.audio_type == consts.bonafide:
                dataset_split = self._normalize_dataset_format(
                    ds=dataset_split, split=config_loader.get_current_split()
                )
                logger.debug("Normalized dataset format for bonafide audio")

            segs_metadata, waves_segs = audio_segmentator.transform(dataset_split)
            logger.info(f"✓ Segmented into {segs_metadata.shape[0]} segments")

            modified_segs_metadata = metadata_modifier.transform(
                current_config=config_loader.get_current_config(),
                metadata=segs_metadata,
            )
            logger.info(f"✓ Modified metadata ({len(modified_segs_metadata.columns)} columns)")

            embeddings = feature_extractor.transform(wave_segments=waves_segs)
            logger.info(f"✓ Extracted {len(embeddings)} embeddings")

            collector.transform(meta_df=modified_segs_metadata, embeddings=embeddings)
            logger.info(f"✓ Saved to {file_name}\n")

    def preprocess_dataset_wavlm(self, file_name=consts.wavlm_file_name_prefix, batch_size=8):
        self._preprocess_dataset(
            file_name=file_name,
            feature_extractor=WavLmExtractor(batch_size=batch_size),
            batch_size=batch_size,
        )

    def preprocess_dataset_fft(self, file_name=consts.fft_file_name_prefix, batch_size=8):
        self._preprocess_dataset(
            file_name=file_name,
            feature_extractor=FFTExtractor(batch_size=batch_size),
            batch_size=batch_size,
        )

    def split_and_balance_dataset(self, file_name=consts.wavlm_file_name_prefix):
        pass
