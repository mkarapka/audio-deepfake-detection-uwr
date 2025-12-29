import torch

from src.common.constants import Constants as consts
from src.pipelines.preprocessing_pipeline import PreprocessingPipeline

if __name__ == "__main__":
    pipeline = PreprocessingPipeline(audio_type=consts.bonafide, config_lst=[None])

    if torch.cuda.is_available():
        batch_size = consts.A_100_BATCH_SIZE
    else:
        batch_size = 8

    pipeline.preprocess_data_set(batch_size=batch_size)
