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

    def get_current_config(self):
        return self.current_config

    def get_current_split(self):
        return self.current_split

    def stream_next_config_dataset(self):
        for config in self.config_lst:
            self.current_config = config
            for split in self.included_splits:
                self.current_split = split
                yield load_audio_dataset_by_streaming(self.source_dataset, config, split)
