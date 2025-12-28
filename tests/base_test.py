from src.common.basic_functions import load_audeter_ds_using_streaming


class BaseTest:
    def __init__(self, split="dev", config="mls-tts-bark", records_no=5):
        self.dataset = load_audeter_ds_using_streaming(
            split=split,
            config=config,
        )
        self.dataset = [record for _, record in zip(range(records_no), self.dataset)]
