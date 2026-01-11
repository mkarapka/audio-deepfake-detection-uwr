import pandas as pd

from src.common.basic_functions import get_logger, setup_logger
from src.common.constants import Constants as consts
from src.preprocessing.data_balancers.mix_blancer import MixBalancer
from src.preprocessing.data_balancers.oversample_real_balancer import (
    OversampleRealBalancer,
)
from src.preprocessing.data_balancers.undersample_spoof_balancer import (
    UndersampleSpoofBalancer,
)
from src.preprocessing.feature_loader import FeatureLoader
from src.training.logistic_regression_trainer import LogisticRegressionTrainer


class BestBalancePipeline:
    def __init__(self):
        self.trained_models = {}
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
        else:
            self.logger.error(f"Unknown balancer type: {balancer_type}")
            return None

    def _train_clf_on_resampled_data(self, oversampling_method: str):
        for ratio in consts.ratios_config[oversampling_method]:
            self.logger.info(
                f"Training Logistic Regression with {oversampling_method} and ratio: {ratio}"
            )
            data_balancer = self._get_balancer_instance(balancer_type=oversampling_method, ratio_args=ratio)
            balanced_metadata = data_balancer.transform(metadata=self.train_split)
            train_embeddings = self.feature_loader.load_embeddings_from_metadata(balanced_metadata)
            dev_embeddings = self.feature_loader.load_embeddings_from_metadata(self.dev_split)

            clf = LogisticRegressionTrainer(
                X_train=train_embeddings,
                y_train=balanced_metadata["target"],
                X_dev=dev_embeddings,
                y_dev=self.dev_split["target"],
            )
            clf.train()
            best_clf = clf.get_best_model()

            ratio_str = f"{ratio[0]}_{ratio[1]}" if oversampling_method == "mix" else f"{ratio}"
            self.trained_models[f"{oversampling_method}{ratio_str}"] = best_clf
        return self.trained_models

    def train_all_balancers(self):
        for balancer_type in consts.ratios_config.keys():
            self.logger.info(f"Training models with balancer: {balancer_type}")
            self._train_clf_on_resampled_data(oversampling_method=balancer_type)

    def pick_best_model(self):
        results = []
        for model_key, model in self.trained_models.items():
            dev_embeddings = self.feature_loader.load_embeddings_from_metadata(self.dev_split)
            accuracy = model.evaluate(X_dev=dev_embeddings, y_dev=self.dev_split["target"])
            results.append((model_key, accuracy))

        results.sort(key=lambda x: x[1], reverse=True)
        results_df = pd.DataFrame(results, columns=["model_key", "accuracy"])
        best_model_key, best_accuracy = results[0]
        self.logger.info(f"Best model: {best_model_key} with accuracy: {best_accuracy:.4f}")
        self.logger.info(f"All results:\n{results_df}")
        return best_model_key, best_accuracy
