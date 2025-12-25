import logging
from pathlib import Path
import pandas as pd
from src.common.constants import Constants as consts

logger = logging.getLogger("audio_deepfake.collector")


class Collector:
    def __init__(self, save_file_path: str):

        self.data_dir = consts.data_dir / "collected_data"
        if self.data_dir.exists() is False:
            self.data_dir.mkdir(parents=True, exist_ok=True)

        if save_file_path is not None:
            self.save_file_path = self.data_dir / Path(save_file_path)
        else:
            logging.info("No save file path provided; data will not be saved to disk.")

    def transform(self, data: pd.DataFrame):
        if self.save_file_path.exists() is True:
            data.to_csv(self.save_file_path, mode="a", index=False, header=False)
        else:
            data.to_csv(self.save_file_path, index=False, header=True)
