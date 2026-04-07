import numpy as np
import pandas as pd

from src.common.constants import Constants as consts
from src.preprocessing.experiment_preprocessor import ExperimentPreprocessor
from src.preprocessing.io.feature_loader import FeatureLoader


def generate_split(size):
    targets = ["bonafide", "spoof"]
    metadata = pd.DataFrame(
        {
            "unique_audio_id": range(size),
            "target": np.random.choice(targets, size=size, p=[0.2, 0.8]),
        }
    )
    features = np.random.rand(size, 10)  # 10 features for each sample
    return metadata, features


TRAIN_SIZE, DEV_SIZE, TEST_SIZE = 100, 70, 70


class FeatureLoaderMock:
    def load_data_split(self, split_name):
        # Return dummy metadata and features based on split name
        if split_name == "train":
            return generate_split(TRAIN_SIZE)
        elif split_name == "dev":
            return generate_split(DEV_SIZE)
        elif split_name == "test":
            return generate_split(TEST_SIZE)
        else:
            raise ValueError(f"Unknown split name: {split_name}")

    def sample_data(self, metadata, features, fraction=0.4):
        return FeatureLoader().sample_data(metadata, features, fraction)

    def sample_by_audio_ids(self, metadata, features, fraction=0.4):
        return FeatureLoader().sample_by_audio_ids(metadata, features, fraction)


class ExperimentPreprocessorTest:
    def __init__(self):
        self.pipeline = ExperimentPreprocessor(
            load_file_name=consts.feature_extracted,
            save_file_name=consts.feature_extracted,
            feat_suffix="",
        )

    def test_get_balancer_instance(self):
        pipeline = ExperimentPreprocessor(
            load_file_name=consts.feature_extracted,
            save_file_name=consts.feature_extracted,
            feat_suffix="",
        )
        balancer = pipeline._get_balancer_instance("undersample", 0.5)
        assert balancer is not None
        assert balancer.real_to_spoof_ratio == 0.5

        balancer = pipeline._get_balancer_instance("oversample", 0.75)
        assert balancer is not None
        assert balancer.real_to_spoof_ratio == 0.75

        balancer = pipeline._get_balancer_instance("mix", [0.5, 1.0])
        assert balancer is not None
        assert balancer.undersample_ratio == 0.5
        assert balancer.oversample_ratio == 1.0

        balancer = pipeline._get_balancer_instance("unbalanced", None)
        assert balancer is None

    def test_prepare_data_for_experiment(self):
        for enabled_audio_ids_sampling in [False, True]:
            undersample_ratio = 1.0
            args_ratio = [undersample_ratio] * 3

            self.pipeline.feature_loader = FeatureLoaderMock()

            splits_config = ("undersample", undersample_ratio)

            preprocess_config = {
                "splits_names": ["train", "dev", "test"],
                "fraction": 1.0,
                "use_audio_id_sampling": enabled_audio_ids_sampling,
                "balance_splits_strategy": splits_config,
            }

            data_for_exp = self.pipeline.preprocess_data(**preprocess_config)

            for i, (_, (meta, feat)) in enumerate(data_for_exp.items()):
                assert isinstance(meta, pd.DataFrame)
                assert isinstance(feat, np.ndarray)
                assert len(meta) == len(feat)
                bonafide_count = meta[meta["target"] == "bonafide"].shape[0]
                if args_ratio[i] is not None:
                    assert args_ratio[i] == bonafide_count / (
                        meta.shape[0] - bonafide_count
                    ), f"Expected ratio: {args_ratio[i]}, got: {bonafide_count / (meta.shape[0] - bonafide_count)}"
                    assert (meta.shape[0] - bonafide_count) > 0  # Ensure there are spoof samples

    def test_convert_labels_to_ints(self):
        y = np.array(["fake", "real", "fake", "real"])
        pos_label = "real"
        expected = np.array([0, 1, 0, 1])
        converted = self.pipeline._convert_labels_to_ints(y, pos_label)
        assert np.all(converted == expected), f"Expected {expected}, got {converted}"


ExperimentPreprocessorTest().test_get_balancer_instance()
ExperimentPreprocessorTest().test_prepare_data_for_experiment()
ExperimentPreprocessorTest().test_convert_labels_to_ints()
