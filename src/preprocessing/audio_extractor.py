from abc import ABC, abstractmethod

class AudioExtractor(ABC):
    @abstractmethod
    def transform(self, data):
        pass