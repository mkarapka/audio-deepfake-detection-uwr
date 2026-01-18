import pandas as pd

from src.common.basic_functions import setup_logger
from src.common.constants import Constants as consts
from src.preprocessing.collector import Collector
from src.preprocessing.data_balancers.mix_blancer import MixBalancer
from src.preprocessing.data_balancers.oversample_real_balancer import (
    OversampleRealBalancer,
)
from src.preprocessing.data_balancers.undersample_spoof_balancer import (
    UndersampleSpoofBalancer,
)
from src.preprocessing.feature_loader import FeatureLoader
from src.training.fft_baseline_classifier import FFTBaselineClassifier
from src.training.record_iterator import RecordIterator


class BestBalancePipeline:
    def __init__(self, RATIOS_CONFIG=consts.ratios_config, objective="f1", is_partial=False):
        self.trained_models = {}
        self.RATIOS_CONFIG = RATIOS_CONFIG
        self.objective = objective
        self.logger = setup_logger(__class__.__name__, log_to_console=True)
        self.feature_loader = FeatureLoader(file_name=consts.feature_extracted)
        self.train_split = self.feature_loader.load_split_file(split_name="train")
        self.dev_split = self.feature_loader.load_split_file(split_name="dev")
        self.is_partial = is_partial

    def _get_balancer_instance(self, balancer_type: str, ratio_args):
        if balancer_type == "undersample":
            return UndersampleSpoofBalancer(real_to_spoof_ratio=ratio_args)
        elif balancer_type == "oversample":
            return OversampleRealBalancer(real_to_spoof_ratio=ratio_args)
        elif balancer_type == "mix":
            undersample_ratio, oversample_ratio = ratio_args
            return MixBalancer(undersample_ratio=undersample_ratio, oversample_ratio=oversample_ratio)
        elif balancer_type == "unbalanced":
            return "unbalanced"
        else:
            self.logger.error(f"Unknown balancer type: {balancer_type}")
            return None

    def _train_clf_on_resampled_data(
        self,
        oversampling_method: str,
        train_split: pd.DataFrame,
        dev_split: pd.DataFrame,
        max_iter=150,
        n_trials=10,
    ):
        for ratio in self.RATIOS_CONFIG[oversampling_method]:
            self.logger.info(f"Training FFT Baseline Classifier with {oversampling_method} and ratio: {ratio}")
            data_balancer = self._get_balancer_instance(balancer_type=oversampling_method, ratio_args=ratio)
            if data_balancer == "unbalanced":
                balanced_train_split = train_split
            else:
                balanced_train_split = data_balancer.transform(metadata=train_split)

            train_embeddings = self.feature_loader.load_embeddings_from_metadata(balanced_train_split)
            dev_embeddings = self.feature_loader.load_embeddings_from_metadata(dev_split)

            clf = FFTBaselineClassifier(
                X_train=train_embeddings,
                y_train=balanced_train_split["target"],
                X_dev=dev_embeddings,
                meta_dev=dev_split,
            )
            clf.train(max_iter=max_iter, n_trials=n_trials, is_partial=self.is_partial)

            ratio_str = f"{ratio[0]}_{ratio[1]}" if oversampling_method == "mix" else f"{ratio}"
            self.trained_models[f"{oversampling_method}{ratio_str}"] = (clf.get_best_value(), clf.get_best_params())

    def _sample_fraction_from_split(self, split: pd.DataFrame, fraction=0.4) -> pd.DataFrame:
        half_size = int(len(split) * fraction)
        reduced_split = split.sample(n=half_size, random_state=42)
        return reduced_split

    def _sample_fraction_uq_audios_from_split(self, split: pd.DataFrame, fraction=0.4) -> pd.DataFrame:
        record_iterator = RecordIterator()
        reduced_split = record_iterator.sample_fraction(metadata=split, fraction=fraction)
        return reduced_split

    def train_all_balancers(self, reduce_factor=0.4, max_iter=200, n_trials=20):
        if self.is_partial:
            self.logger.info("Using unique audio ids to reduce dataset for partial training.")
            reduced_train_split = self._sample_fraction_uq_audios_from_split(self.train_split, fraction=reduce_factor)
            reduced_dev_split = self._sample_fraction_uq_audios_from_split(self.dev_split, fraction=reduce_factor)
        else:
            self.logger.info("Using random sampling to reduce dataset for partial training.")
            reduced_train_split = self._sample_fraction_from_split(self.train_split, fraction=reduce_factor)
            reduced_dev_split = self._sample_fraction_from_split(self.dev_split, fraction=reduce_factor)

        for balancer_type in self.RATIOS_CONFIG.keys():
            self.logger.info(
                f"Training models with balancer: {balancer_type},and reduced size by {
                    len(reduced_train_split)
                }/{len(self.train_split):.2f}"
            )
            self._train_clf_on_resampled_data(
                oversampling_method=balancer_type,
                train_split=reduced_train_split,
                dev_split=reduced_dev_split,
                max_iter=max_iter,
                n_trials=n_trials,
            )

    def create_dataframe_and_save(self, file_name: str):
        df = {"model_name": [], f"{self.objective}": [], "best_params": []}
        for model_name, (objective, params) in self.trained_models.items():
            self.logger.info(f"Model: {model_name}, Recall: {objective}, Params: {params}")
            df["model_name"].append(model_name)
            df[f"{self.objective}"].append(objective)
            df["best_params"].append(params)
        results_df = pd.DataFrame(df)
        collector = Collector(save_file_name=file_name)
        collector._write_data_to_csv(results_df, include_index=False)

    def pick_best_model(self):
        best_model_name = None
        best_objective = -1.0

        for model_name, (objective, params) in self.trained_models.items():
            self.logger.info(f"Model: {model_name}, Recall: {objective}, Params: {params}")
            if objective > best_objective:
                best_objective = objective
                best_model_name = model_name

        self.logger.info(f"Best model is {best_model_name} with Recall: {best_objective}")
        return best_model_name, best_objective
