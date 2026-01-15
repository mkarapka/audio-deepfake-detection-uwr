from abc import ABC, abstractmethod

from src.common.logger import get_logger


class BasePreprocessor(ABC):
    def __init__(self, class_name=None):
        self.class_name = class_name
        if self.class_name:
            self.logger = get_logger(f"audio_deepfake.{self.class_name}")
        else:
            self.logger = get_logger("audio_deepfake.base_preprocessor")
        self.logger.info(f"Initialized preprocessor: {self.class_name}")

    @abstractmethod
    def transform(self, data):
        pass
