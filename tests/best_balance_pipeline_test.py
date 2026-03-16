import pandas as pd
from sklearn.dummy import DummyClassifier

from src.common.constants import BalanceType as bt
from src.common.constants import Constants as consts
from src.common.constants import ExperimentPreprocessConfig
from src.models.base_model import BaseModel
from src.pipelines.experiments.best_balance_pipeline import BestBalancePipeline


class DummyModel(BaseModel):
    def __init__(self):
        super().__init__(class_name="DummyModel")
        self.model = DummyClassifier(strategy="most_frequent")

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


SIZE = 100


def dummy_objective(trial, model, X_train, y_train, X_dev, y_dev):
    trial.suggest_categorical("strategy", ["most_frequent", "stratified", "uniform"])
    model.fit(X_train, y_train)
    preds = model.predict(X_dev)
    accuracy = (preds == y_dev).mean()
    return accuracy


EXPECTED_COLUMN_NAMES = [
    "model_name",
    "splits_balance_configs",
    "n_trials",
    "parameters",
    "metrics_names",
    "metrics_scores",
]


class TestBestBalancePipeline:
    def TestRunPipeline():
        preprocess_config = {
            "splits_names": ["train", "dev"],
            "fraction": 0.05,
            "is_audio_ids_sampling": False,
            "balance_configs": None,
        }

        balance_configs_for_experiment = {
            bt.UNDERSAMPLE: [0.5, 0.75, 1.0],
            bt.OVERSAMPLE: [0.5, 0.75, 1.0],
            bt.MIX: [[0.5, 1.0]],
            bt.UNBALANCED: [None],
        }
        experiment_config = {
            "clf_model": DummyModel(),
            "objective": dummy_objective,
            "preprocess_config": preprocess_config,
            "balances_configs": balance_configs_for_experiment,
        }

        pipeline = BestBalancePipeline(**experiment_config)
        pipeline.run()

        expected_results = pd.DataFrame(
            {
                "model_name": ["DummyModel"],
                "splits_balance_configs": [[[bt.MIX, 0.7], [bt.UNBALANCED, 0.7]]],
                "n_trials": [20],
                "parameters": None,
                "metrics_names": ["accuracy"],
                "metrics_scores": None,
            }
        )
        read_results = pd.read_csv(consts.train_results_dir / consts.best_balance_results)

        assert (
            read_results.columns.tolist() == EXPECTED_COLUMN_NAMES
        ), f"Expected columns: {EXPECTED_COLUMN_NAMES}, but got: {read_results.columns.tolist()}"
        assert read_results.get("metrics_scores").all().dtype is float, "Expected 'metrics_scores' column in results."
        results_to_compare = read_results.drop(columns=["metrics_scores", "parameters"])
        assert results_to_compare.equals(
            expected_results.drop(columns=["metrics_scores", "parameters"])
        ), "Results do not match expected values."


TestBestBalancePipeline.TestRunPipeline()
