from abc import ABC, abstractmethod

from src.common.logger import setup_logger


class BasePreprocessor(ABC):
    def __init__(self, class_name=None):
        self.class_name = class_name
        if self.class_name:
            self.logger = setup_logger(self.class_name, log_to_console=True)
        else:
            self.logger = setup_logger(self.class_name, log_to_console=True)
        self.logger.info(f"Initialized preprocessor: {self.class_name}")

    @abstractmethod
    def transform(self, data):
        pass
