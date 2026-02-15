from src.common.basic_functions import get_batch_size
from src.common.constants import Constants as consts
from src.pipelines.preprocessing.preprocessing_pipeline import PreprocessingPipeline

if __name__ == "__main__":
    batch_size = get_batch_size()

    pipeline_bonafide = PreprocessingPipeline(consts.bonafide, config_lst=[None])
    pipeline_bonafide.preprocess_dataset_wavlm(batch_size=batch_size)

    pipeline_spoof = PreprocessingPipeline(consts.spoof, config_lst=consts.tts_and_vocoders_configs)
    pipeline_spoof.preprocess_dataset_wavlm(batch_size=batch_size)
