from src.common.basic_functions import load_audeter_ds_using_streaming
from src.common.constants import Constants as consts


class ConfigLoader:
    def __init__(self, config: list[str], splits: list[str] = ["dev", "test"]):
        self.config_lst = config
        self.current_dataset = None
        self.included_splits = splits

        if consts.data_dir.exists() is False:
            consts.data_dir.mkdir(parents=True, exist_ok=True)

    def load_next_config(self):
        for config in self.config_lst:
            for split in self.included_splits:
                self.current_dataset = load_audeter_ds_using_streaming(config, split)
                yield self.current_dataset, config, split