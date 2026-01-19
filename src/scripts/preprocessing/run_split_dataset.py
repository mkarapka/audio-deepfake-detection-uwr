from src.common.constants import Constants as consts
from src.pipelines.preprocessing_pipeline import PreprocessingPipeline

if __name__ == "__main__":
    pipeline = PreprocessingPipeline(audio_type=consts.bonafide)

    pipeline.split_dataset(
        file_name=consts.feature_extracted,
        split_config=consts.basic_train_dev_test_config,
        seed=44,
    )
