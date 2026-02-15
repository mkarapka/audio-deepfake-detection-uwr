from src.common.basic_functions import generate_config_sample, get_batch_size
from src.common.constants import Constants as consts
from src.pipelines.preprocessing.preprocessing_pipeline import PreprocessingPipeline

if __name__ == "__main__":
    configs_lst = generate_config_sample(seed=42, num_tts=2, num_vocoders=2)
    pipeline = PreprocessingPipeline(consts.spoof, config_lst=configs_lst)
    batch_size = get_batch_size()
    pipeline.preprocess_dataset_fft(batch_size=batch_size)
