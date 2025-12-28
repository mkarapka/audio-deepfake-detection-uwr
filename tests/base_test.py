from src.common.basic_functions import load_audeter_ds_using_streaming
from src.common.logger import get_logger


class BaseTest:
    def __init__(self, split="dev", config="mls-tts-bark", records_no=2):
        self.logger = get_logger()
        self.dataset = load_audeter_ds_using_streaming(
            split=split,
            config=config,
        )
        self.dataset = [record for _, record in zip(range(records_no), self.dataset)]
