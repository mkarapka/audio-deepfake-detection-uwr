from src.common.basic_functions import get_batch_size
from src.common.constants import Constants as consts
from src.pipelines.preprocessing_pipeline import PreprocessingPipeline

if __name__ == "__main__":
    pipeline = PreprocessingPipeline(audio_type=consts.bonafide, config_lst=[None])
    batch_size = get_batch_size()
    pipeline.preprocess_dataset(batch_size=batch_size)
