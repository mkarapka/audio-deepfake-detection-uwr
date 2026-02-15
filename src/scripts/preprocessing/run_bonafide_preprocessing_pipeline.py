from src.common.basic_functions import get_batch_size
from src.common.constants import Constants as consts
from src.pipelines.preprocessing.preprocessing_pipeline import PreprocessingPipeline

if __name__ == "__main__":
    pipeline = PreprocessingPipeline(audio_type=consts.bonafide, config_lst=[None])
    batch_size = get_batch_size()
    pipeline.preprocess_dataset_wavlm(batch_size=batch_size)
