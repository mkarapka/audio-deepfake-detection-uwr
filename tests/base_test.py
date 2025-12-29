from src.common.basic_functions import load_audio_dataset_by_streaming
from src.common.constants import Constants as consts
from src.common.logger import get_logger


class BaseTest:
    def __init__(
        self,
        source_dataset=consts.audeter_ds_path,
        split="dev",
        config="mls-tts-bark",
        records_no=2,
    ):
        self.logger = get_logger()
        self.dataset = load_audio_dataset_by_streaming(
            dataset=source_dataset,
            split=split,
            config=config,
        )
        self.dataset = [record for _, record in zip(range(records_no), self.dataset)]
