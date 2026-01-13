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
from src.training.logistic_regression_classifier import LogisticRegressionClassifier


class BestBalancePipeline:
    def __init__(self, RATIOS_CONFIG=consts.ratios_config, objective="f1"):
        self.trained_models = {}
        self.RATIOS_CONFIG = RATIOS_CONFIG
        self.objective = objective
        self.logger = setup_logger(__class__.__name__, log_to_console=True)
        self.feature_loader = FeatureLoader(file_name=consts.feature_extracted)
        self.train_split = self.feature_loader.load_split_file(split_name="train")
        self.dev_split = self.feature_loader.load_split_file(split_name="dev")

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
            self.logger.info(f"Training Logistic Regression with {oversampling_method} and ratio: {ratio}")
            data_balancer = self._get_balancer_instance(balancer_type=oversampling_method, ratio_args=ratio)
            if data_balancer == "unbalanced":
                balanced_train_split = train_split
            else:
                balanced_train_split = data_balancer.transform(metadata=train_split)

            train_embeddings = self.feature_loader.load_embeddings_from_metadata(balanced_train_split)
            dev_embeddings = self.feature_loader.load_embeddings_from_metadata(dev_split)

            clf = LogisticRegressionClassifier(
                X_train=train_embeddings,
                y_train=balanced_train_split["target"],
                X_dev=dev_embeddings,
                y_dev=dev_split["target"],
                objective=self.objective,
            )
            clf.train(max_iter=max_iter, n_trials=n_trials)

            ratio_str = f"{ratio[0]}_{ratio[1]}" if oversampling_method == "mix" else f"{ratio}"
            self.trained_models[f"{oversampling_method}{ratio_str}"] = (clf.get_best_value(), clf.get_best_params())

    def _sample_fraction_from_split(self, split: pd.DataFrame, fraction=0.4) -> pd.DataFrame:
        half_size = int(len(split) * fraction)
        reduced_split = split.sample(n=half_size, random_state=42)
        return reduced_split

    def train_all_balancers(self, reduce_factor=0.4, max_iter=200, n_trials=20):
        reduced_train_split = self._sample_fraction_from_split(self.train_split, fraction=reduce_factor)
        reduced_dev_split = self._sample_fraction_from_split(self.dev_split, fraction=reduce_factor)

        for balancer_type in self.RATIOS_CONFIG.keys():
            self.logger.info(
                f"Training models with balancer: {balancer_type}, and reduced size by {len(reduced_train_split)}/{len(self.train_split):.2f}"
            )
            self._train_clf_on_resampled_data(
                oversampling_method=balancer_type,
                train_split=reduced_train_split,
                dev_split=reduced_dev_split,
                max_iter=max_iter,
                n_trials=n_trials,
            )

    def create_dataframe_and_save(self, file_name: str):
        df = {"model_name": [], "recall": [], "best_params": []}
        for model_name, (recall, params) in self.trained_models.items():
            self.logger.info(f"Model: {model_name}, Recall: {recall}, Params: {params}")
            df["model_name"].append(model_name)
            df["recall"].append(recall)
            df["best_params"].append(params)
        results_df = pd.DataFrame(df)
        collector = Collector(save_file_name=file_name)
        collector._write_data_to_csv(results_df, include_index=False)

    def pick_best_model(self):
        best_model_name = None
        best_recall = -1.0

        for model_name, (recall, params) in self.trained_models.items():
            self.logger.info(f"Model: {model_name}, Recall: {recall}, Params: {params}")
            if recall > best_recall:
                best_recall = recall
                best_model_name = model_name

        self.logger.info(f"Best model is {best_model_name} with Recall: {best_recall}")
        return best_model_name, best_recall
