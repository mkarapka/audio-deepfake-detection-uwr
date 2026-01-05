from src.pipelines.preprocessing_pipeline import PreprocessingPipeline
from src.common.constants import Constants as consts

if __name__ == "__main__":
    pipeline = PreprocessingPipeline(audio_type=consts.bonafide)

    pipeline.split_dataset(
        file_name=consts.wavlm_file_name_prefix,
        split_config=consts.basic_train_dev_test_config,
        seed=44,
    )