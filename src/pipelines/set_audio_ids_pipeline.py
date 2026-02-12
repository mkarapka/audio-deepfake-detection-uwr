from src.common.basic_functions import setup_logger
from src.common.constants import Constants as consts
from src.preprocessing.io.collector import Collector
from src.preprocessing.io.feature_loader import FeatureLoader
from src.preprocessing.unique_audio_id_mapper import UniqueAudioIdMapper


class SetAudioIDsPipeline:
    def __init__(self, output_file: str):
        self.SPLITS = ["train", "dev", "test"]
        self.output_file = output_file
        self.logger = setup_logger(__class__.__name__, log_to_console=True)

        self.feature_loader = FeatureLoader(file_name=consts.feature_extracted)
        self.uq_audio_id_mapper = UniqueAudioIdMapper()
        self.collector = Collector(save_file_name=self.output_file)

    def _set_unique_audio_ids(self, split_name: str):
        self.logger.info(f"Loading metadata for split: {split_name}")
        metadata = self.feature_loader.load_meta_split(split_name=split_name)

        self.logger.info("Setting unique audio IDs")
        modified_metadata = self.uq_audio_id_mapper.transform(metadata=metadata)

        self.logger.info(f"Unique audio IDs set for split: {split_name}")
        return modified_metadata

    def _set_unique_audio_ids_for_main_metadata(self):
        self.logger.info("Loading main metadata")
        metadata = self.feature_loader.load_metadata_file()

        self.logger.info("Setting unique audio IDs for main metadata")
        modified_metadata = self.uq_audio_id_mapper.transform(metadata=metadata)

        self.logger.info("Unique audio IDs set for main metadata")
        return modified_metadata

    def run(self):
        data = []
        for split_name in self.SPLITS:
            modified_metadata = self._set_unique_audio_ids(split_name=split_name)
            data.append(modified_metadata)

        self.logger.info("SetAudioIDsPipeline completed, saving results.")
        self.collector.transform_splits(data=data, splits=self.SPLITS)

    def run_on_main_metadata(self):
        modified_metadata = self._set_unique_audio_ids_for_main_metadata()

        self.logger.info("SetAudioIDsPipeline on main metadata completed, saving results.")
        self.collector.write_data_to_csv(data=modified_metadata)
