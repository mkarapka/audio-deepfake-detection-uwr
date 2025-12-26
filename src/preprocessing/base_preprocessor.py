from abc import ABC, abstractmethod

from src.common.logger import get_logger


class BasePreprocessor(ABC):
    def __init__(self, class_name=None):
        if class_name:
            self.logger = get_logger(f"audio_deepfake.{class_name}")
        else:
            self.logger = get_logger("audio_deepfake.base_preprocessor")
        self.logger.info(f"Initialized preprocessor: {self.__class__.__name__}")

    @abstractmethod
    def transform(self, data):
        pass
