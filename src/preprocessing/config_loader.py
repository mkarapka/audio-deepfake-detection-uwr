from src.common.basic_functions import load_audeter_ds_using_streaming
from src.common.constants import Constants as consts


class ConfigLoader:
    def __init__(self, config: list[str], splits: list[str] = ["dev", "test"]):
        self.current_dataset = None
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
                self.current_dataset = load_audeter_ds_using_streaming(config, split)
                yield self.current_dataset
