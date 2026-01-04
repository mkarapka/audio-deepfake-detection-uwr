from src.preprocessing.base_preprocessor import BasePreprocessor


class BaseFeatureExtractor(BasePreprocessor):
    def __init__(self, class_name: str):
        super().__init__(class_name=class_name)

    def transform(self, data: any) -> any:
        raise NotImplementedError("Transform method must be implemented in the subclass.")
