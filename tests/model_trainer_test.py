import joblib
import numpy as np
import pandas as pd

from src.common.basic_functions import print_green
from src.common.constants import Constants as consts
from src.models.model_trainer import ModelTrainer


class ModelTrainerTest:
    def __init__(self):
        self.trainer = ModelTrainer()

    def test_get_best_params_without_training(self):
        best_params = self.trainer.get_best_params()
        assert best_params is None, "Expected None when no training has been done."

    def test_convert_labels_to_ints(self):
        y = np.array(["fake", "real", "fake", "real"])
        pos_label = "real"
        expected = np.array([0, 1, 0, 1])
        converted = self.trainer._convert_labels_to_ints(y, pos_label)
        assert np.all(converted == expected), f"Expected {expected}, got {converted}"

    def test_optuna_train(self):
        def dummy_objective(trial, model, **params):
            return trial.suggest_float("x", 0, 1)

        class DummyModel:
            pass

        self.trainer.optuna_train(model=DummyModel(), objective=dummy_objective, n_trials=10)
        best_params = self.trainer.get_best_params()
        assert "x" in best_params, "Expected 'x' in best parameters."

    def test_save_results(self):
        FILE_NAME = "test_results.csv"

        attributes = ["model_name", "n_trials", "data_type", "balance_type", "score"]
        values = ["logistic_regression", 50, "fft_feat", "mix_balanced_0.5_1.0", 80.000001212112]
        params = {a: v for a, v in zip(attributes, values)}
        self.trainer.save_results(FILE_NAME, params)

        saved_df = pd.read_csv(consts.train_results_dir / FILE_NAME)
        for attr in attributes:
            assert attr in saved_df.columns, f"Expected column '{attr}' in saved results."
            assert (
                saved_df[attr].iloc[-1] == params[attr]
            ), f"Expected value '{params[attr]}' for column '{attr}', got '{saved_df[attr].iloc[-1]}'"

    def test_get_target(self):
        example_metadata = pd.DataFrame(
            {"target": ["bonafide", "spoof", "bonafide", "spoof"], "other_info": [1, 2, 3, 4]}
        )

        y = self.trainer.get_target(example_metadata, pos_label="bonafide")
        expected_y = np.array([1, 0, 1, 0])  # bonafide -> 1, spoof -> 0

        assert np.all(y == expected_y), f"Expected y {expected_y}, got {y}"

    def test_save_model(self):
        DUMMY_CONTENT = "dummy model content"

        class DummyModel:
            def save(self, file_path):
                joblib.dump(DUMMY_CONTENT, file_path)

        dummy_model = DummyModel()
        FILE_NAME = "test_dummy_model"
        self.trainer.save_model(dummy_model, FILE_NAME, ext="txt")

        saved_files = list(consts.models_dir.glob(f"{FILE_NAME}*.txt"))
        assert len(saved_files) > 0, "Expected at least one saved model file."
        with open(saved_files[0], "rb") as f:
            content = joblib.load(f)
            assert content == DUMMY_CONTENT, f"Expected '{DUMMY_CONTENT}', got '{content}'"


ModelTrainerTest().test_get_best_params_without_training()
ModelTrainerTest().test_convert_labels_to_ints()
ModelTrainerTest().test_optuna_train()
ModelTrainerTest().test_save_results()
ModelTrainerTest().test_get_target()
ModelTrainerTest().test_save_model()
print_green("All ModelTrainer tests passed successfully!")
