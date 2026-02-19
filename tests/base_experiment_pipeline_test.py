import numpy as np
import pandas as pd

from src.common.constants import BalanceType, SplitConfig
from src.pipelines.experiments.base_experiment_pipeline import BaseExperimentPipeline


def generate_split(size):
    targets = ["bonafide", "spoof"]
    metadata = pd.DataFrame(
        {
            "audio_id": range(size),
            "target": np.random.choice(targets, size=size, p=[0.2, 0.8]),
        }
    )
    features = np.random.rand(size, 10)  # 10 features for each sample
    return metadata, features


class FeatureLoaderMock:
    def load_data_split(self, split_name):
        # Return dummy metadata and features based on split name
        if split_name == "train":
            return generate_split(100)
        elif split_name == "dev":
            return generate_split(70)
        elif split_name == "test":
            return generate_split(70)
        else:
            raise ValueError(f"Unknown split name: {split_name}")


class BaseExperimentPipelineTest:
    def test_get_balancer_instance(self):
        pipeline = BaseExperimentPipeline(splits_config={}, feat_suffix="")
        balancer = pipeline._get_balancer_instance(BalanceType.UNDERSAMPLE, 0.5)
        assert balancer is not None
        assert balancer.real_to_spoof_ratio == 0.5

        balancer = pipeline._get_balancer_instance(BalanceType.OVERSAMPLE, 0.75)
        assert balancer is not None
        assert balancer.real_to_spoof_ratio == 0.75

        balancer = pipeline._get_balancer_instance(BalanceType.MIX, [0.5, 1.0])
        assert balancer is not None
        assert balancer.undersample_ratio == 0.5
        assert balancer.oversample_ratio == 1.0

        balancer = pipeline._get_balancer_instance(BalanceType.UNBALANCED, None)
        assert balancer is None

    def test_prepare_data_for_experiment(self):
        args_ratio = [0.5, 1.0, None]
        splits_config = {
            "train": SplitConfig(balance_type=BalanceType.UNDERSAMPLE, ratio_args=0.5),
            "dev": SplitConfig(balance_type=BalanceType.MIX, ratio_args=[0.5, 1.0]),
            "test": SplitConfig(balance_type=BalanceType.UNBALANCED, ratio_args=None),
        }
        pipeline = BaseExperimentPipeline(splits_config=splits_config, feat_suffix="")
        pipeline.feature_loader = FeatureLoaderMock()  # Use the mock feature loader
        data_for_exp = pipeline.prepare_data_for_experiment()

        for i, (_, (meta, feat)) in enumerate(data_for_exp.items()):
            assert isinstance(meta, pd.DataFrame)
            assert isinstance(feat, np.ndarray)
            assert len(meta) == len(feat)
            bonafide_count = meta[meta["target"] == "bonafide"].shape[0]
            if args_ratio[i] is not None:
                assert args_ratio[i] == bonafide_count / (meta.shape[0] - bonafide_count)


BaseExperimentPipelineTest().test_get_balancer_instance()
BaseExperimentPipelineTest().test_prepare_data_for_experiment()
