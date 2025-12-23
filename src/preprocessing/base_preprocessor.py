from abc import ABC, abstractmethod

class BasePreprocessor(ABC):
    @abstractmethod
    def transform(self, data):
        pass
