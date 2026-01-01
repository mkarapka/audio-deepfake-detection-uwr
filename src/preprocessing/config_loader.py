import pandas as pd

from src.common.basic_functions import load_audio_dataset_by_streaming
from src.common.constants import Constants as consts
from src.common.logger import get_logger, setup_logger

logger = get_logger("ConfigLoader")
setup_logger("ConfigLoader", log_to_console=False)


class ConfigLoader:
    def __init__(self, source_dataset: str, config: list[str], splits: list[str] = ["dev", "test"]):
        if source_dataset is None:
            logger.error("Dataset name must be provided to initialize ConfigLoader.")
        self.source_dataset = source_dataset

        self.current_config = None
        self.current_split = None
        self.config_lst = config
        self.included_splits = splits

        if consts.data_dir.exists() is False:
            consts.data_dir.mkdir(parents=True, exist_ok=True)
              
        if consts.collected_data_dir.exists() is False:
            consts.collected_data_dir.mkdir(parents=True, exist_ok=True)

    def _generate_speakers_ids_map(self) -> pd.DataFrame:
        def get_speaker_id(record):
            return int(record["speaker_id"])

        if self.source_dataset != consts.mls_eng_ds_path or self.config_lst != [None]:
            logger.error("generate_speakers_ids_map is only implemented for bonafide dataset.")
            return

        maps_for_splits = {}
        for dataset_split in self.stream_next_config_dataset():
            speaker_ids = []
            for record in dataset_split:
                speaker_ids.append(get_speaker_id(record))
            maps_for_splits[self.current_split] = pd.Series(speaker_ids)

        df = pd.DataFrame(maps_for_splits).fillna(-1).astype(int)
        return df

    def get_current_config(self):
        if self.current_config is None:
            return consts.mls_eng_config
        return self.current_config

    def get_current_split(self):
        return self.current_split

    def stream_next_config_dataset(self):
        for config in self.config_lst:
            self.current_config = config
            for split in self.included_splits:
                self.current_split = split
                yield load_audio_dataset_by_streaming(self.source_dataset, config, split)

    def load_speakers_ids(self, save_file=consts.speakers_ids_file):
        file_path = consts.collected_data_dir / save_file
        if file_path.exists():
            logger.info(f"Loaded existing speakers IDs from {save_file}")
            return pd.read_csv(file_path)
        else:
            logger.info(f"Generating new speakers IDs and saving to {save_file}")
            speakers_ids_df = self._generate_speakers_ids_map()
            speakers_ids_df.to_csv(file_path, index=False)
            logger.info(f"Saved speakers IDs to {save_file}")
            return speakers_ids_df
