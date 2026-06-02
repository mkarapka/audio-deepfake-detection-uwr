from src.common.utils import get_batch_size
from src.pipelines.preprocessing.in_the_wild_preprocessing_pipeline import (
    InTheWildPreprocessingPipeline,
)

if __name__ == "__main__":
    batch_size = get_batch_size()

    pipeline = InTheWildPreprocessingPipeline()
    pipeline.preprocess_dataset_fft(batch_size=batch_size)
    pipeline.preprocess_dataset_wavlm(batch_size=batch_size)
