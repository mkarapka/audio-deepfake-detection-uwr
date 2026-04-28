import numpy as np
import pandas as pd

from src.common.utils import print_green
from src.common.constants import Constants as consts
from src.preprocessing.experiment_preprocessor import ExperimentPreprocessor
from src.preprocessing.io.feature_loader import FeatureLoader


def generate_split(size):
    targets = ["bonafide", "spoof"]
    metadata = pd.DataFrame(
        {
            "unique_audio_id": range(size),
            "target": np.random.choice(targets, size=size, p=[0.2, 0.8]),
            "subclass": np.random.randint(0, 5, size=size),
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

    def sample_data(self, metadata, features, fraction=0.4, audio_id_sampling=False):
        return FeatureLoader().sample_data(metadata, features, fraction)


class DeterministicFeatureLoaderMock:
    def _build_split(self, size):
        metadata = pd.DataFrame(
            {
                "unique_audio_id": range(size),
                "target": ["bonafide" if i % 2 == 0 else "spoof" for i in range(size)],
                "subclass": [i % 3 for i in range(size)],
            }
        )
        features = np.random.rand(size, 10)
        return metadata, features

    def load_data_split(self, split_name):
        if split_name == "train":
            return self._build_split(TRAIN_SIZE)
        elif split_name == "dev":
            return self._build_split(DEV_SIZE)
        elif split_name == "test":
            return self._build_split(TEST_SIZE)
        raise ValueError(f"Unknown split name: {split_name}")

    def sample_data(self, metadata, features, fraction=0.4, audio_id_sampling=False):
        return FeatureLoader().sample_data(metadata, features, fraction)


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
            undersample_ratio, oversample_ratio, mix_ratio = 1.0, 1.0, [0.5, 1.0]
            args_ratio = [undersample_ratio, oversample_ratio, mix_ratio[-1]]

            self.pipeline.feature_loader = FeatureLoaderMock()

            splits_config = [("undersample", undersample_ratio), ("oversample", oversample_ratio), ("mix", mix_ratio)]

            preprocess_config = {
                "splits_names": ["train", "dev", "test"],
                "fraction": 1.0,
                "use_audio_id_sampling": enabled_audio_ids_sampling,
                "balance_splits_strategy": splits_config,
            }

            data_for_exp = self.pipeline.preprocess_data(**preprocess_config)

            for i, (split_name, loader) in enumerate(data_for_exp.items()):
                meta = loader.metadata
                feat = loader.features

                assert isinstance(meta, pd.DataFrame)
                assert isinstance(feat, np.ndarray)
                assert len(meta) == len(feat)
                bonafide_count = meta[meta["target"] == "bonafide"].shape[0]
                if args_ratio[i] is not None:
                    assert args_ratio[i] == bonafide_count / (
                        meta.shape[0] - bonafide_count
                    ), f"Expected ratio: {args_ratio[i]}, got: {bonafide_count / (meta.shape[0] - bonafide_count)}"
                    assert (meta.shape[0] - bonafide_count) > 0  # Ensure there are spoof samples

    def test_get_target(self):
        metadata = pd.DataFrame(
            {
                "target": ["bonafide", "spoof", "bonafide", "spoof"],
            }
        )
        expected = np.array([1, 0, 1, 0])
        result = self.pipeline.get_target(metadata=metadata, pos_label="bonafide")
        assert np.all(result == expected), f"Expected {expected}, got {result}"

    def test_remove_subclass_from_split(self):
        metadata, features = generate_split(size=50)
        original_size = len(metadata)

        # Remove subclass label 0
        subclass_label_to_remove = 0
        meta_filtered, feat_filtered = self.pipeline._remove_subclass_from_split(
            metadata=metadata, features=features, subclass_label=subclass_label_to_remove
        )

        # Check that subclass label 0 is removed
        assert subclass_label_to_remove not in meta_filtered["subclass"].values
        assert len(meta_filtered) < original_size
        assert len(feat_filtered) == len(meta_filtered)
        assert feat_filtered.shape == (len(meta_filtered), 10)

    def test_remove_records_by_query(self):
        metadata = pd.DataFrame(
            {
                "unique_audio_id": [0, 1, 2, 3],
                "target": ["bonafide", "spoof", "bonafide", "spoof"],
                "subclass": [0, 1, 2, 3],
            }
        )
        features = np.random.rand(4, 10)

        meta_filtered, feat_filtered = self.pipeline._remove_records_by_query(
            metadata=metadata,
            features=features,
            query="target == 'bonafide'",
        )

        assert len(meta_filtered) == 2
        assert (meta_filtered["target"] == "spoof").all()
        assert len(feat_filtered) == len(meta_filtered)

    def test_prepare_data_with_remove_by_query(self):
        self.pipeline.feature_loader = DeterministicFeatureLoaderMock()

        preprocess_config = {
            "splits_names": ["train", "dev", "test"],
            "fraction": 1.0,
            "use_audio_id_sampling": False,
            "balance_splits_strategy": [None, None, None],
            "remove_by_query": "target == 'bonafide'",
        }

        data_for_exp = self.pipeline.preprocess_data(**preprocess_config)

        for _, loader in data_for_exp.items():
            meta = loader.metadata
            feat = loader.features
            assert (meta["target"] == "spoof").all()
            assert len(meta) == len(feat)


ExperimentPreprocessorTest().test_get_balancer_instance()
ExperimentPreprocessorTest().test_prepare_data_for_experiment()
ExperimentPreprocessorTest().test_get_target()
ExperimentPreprocessorTest().test_remove_subclass_from_split()
ExperimentPreprocessorTest().test_remove_records_by_query()
ExperimentPreprocessorTest().test_prepare_data_with_remove_by_query()
print_green("All tests passed!")
