from src.common.constants import Constants as consts
from src.pipelines.best_balance_pipeline import BestBalancePipeline


class TestBestBalancePipeline:
    def test_initialization(self):
        pipeline = BestBalancePipeline(
            RATIOS_CONFIG=consts.only_mix_equal_ratio_config,
            objective="f1",
            is_partial=True,
        )
        assert pipeline.RATIOS_CONFIG == consts.only_mix_equal_ratio_config
        assert pipeline.objective == "f1"
        assert pipeline.is_partial is True
        print("BestBalancePipeline initialization test passed.")

    def test_sample_fraction_uq_audios_from_split(self, is_train, fraction=0.1, check_size=20):
        pipeline = BestBalancePipeline(
            RATIOS_CONFIG=consts.only_mix_equal_ratio_config,
            objective="f1",
            is_partial=True,
        )
        split = pipeline.train_split if is_train else pipeline.dev_split
        sampled_split = pipeline._sample_fraction_uq_audios_from_split(
            split=split, frac=fraction, is_train_split=is_train
        )

        print(
            f"Fraction of metadata after sampling, column 'unique_audio_id': {
                sampled_split['unique_audio_id'].head(10)}"
        )
        if is_train:
            assert sampled_split["unique_audio_id"].iloc[:check_size].nunique() >= int(check_size * 0.8)
        else:
            assert sampled_split["unique_audio_id"].iloc[:check_size].nunique() <= int(check_size * 0.2)

    def test_sample_fraction_from_split_basic(self, is_train, fraction=0.1, check_size=20):
        pipeline = BestBalancePipeline(
            RATIOS_CONFIG=consts.only_mix_equal_ratio_config,
            objective="f1",
            is_partial=True,
        )
        split = pipeline.train_split if is_train else pipeline.dev_split
        sampled_split = pipeline._sample_fraction_from_split_basic(split=split, frac=fraction)

        print(
            f"Fraction of metadata after basic sampling, column 'unique_audio_id': {
                sampled_split['unique_audio_id'].head(10)}"
        )
        assert sampled_split["unique_audio_id"].iloc[:check_size].nunique() >= int(check_size * 0.8)


TestBestBalancePipeline().test_initialization()
print("Train fraction tests:")
TestBestBalancePipeline().test_sample_fraction_uq_audios_from_split(is_train=True, fraction=0.1)
print("Dev fraction tests:")
TestBestBalancePipeline().test_sample_fraction_uq_audios_from_split(is_train=False, fraction=0.1)
print("--------------------------------------------------")
print("Basic sample fraction tests:")
print("Train fraction test:")
TestBestBalancePipeline().test_sample_fraction_from_split_basic(fraction=0.1, is_train=True)
print("Dev fraction test:")
TestBestBalancePipeline().test_sample_fraction_from_split_basic(fraction=0.1, is_train=False)
