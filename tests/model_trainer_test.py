import numpy as np
import pandas as pd

from src.common.basic_functions import print_green
from src.common.constants import Constants as consts
from src.models.base_model import BaseModel
from src.models.model_trainer import ModelTrainer
from src.models.objectives import Objective


class ModelTrainerTest:
    def __init__(self):
        self.trainer = ModelTrainer(show_progress_bar=False)

    def test_get_best_params_without_training(self):
        best_params = self.trainer.get_best_params()
        assert best_params is None, "Expected None when no training has been done."

    def test_optuna_train(self):
        class DummyModel:
            pass

        class DummyObjective(Objective):
            def __init__(self):
                super().__init__(model=DummyModel, direction="maximize")

            def __call__(self, *, trial, **params):
                return trial.suggest_float("x", 0, 1)

        X_train = np.array([[0.0], [1.0]])
        y_train = np.array([0, 1])
        X_dev = np.array([[0.0], [1.0]])
        y_dev = np.array([0, 1])

        objective = DummyObjective()

        self.trainer.optuna_train(
            objective=objective, n_trials=10, X_train=X_train, y_train=y_train, X_val=X_dev, y_val=y_dev
        )
        best_params = self.trainer.get_best_params()
        assert "x" in best_params, "Expected 'x' in best parameters."

        best_value = self.trainer.get_best_value()
        assert best_value is not None, "Expected best value to be set after training."

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

    def test_save_model(self):
        class DummyModel(BaseModel):
            def __init__(self):
                super().__init__(
                    class_name="DummyModel", models_dir=consts.tests_data_dir / "models", include_mps=False
                )

            def load(self, model_name: str, ext: str, sub_dir: str = None):
                return None

            def save(self, model_name: str, ext: str, sub_dir: str = None):
                file_path = self.models_dir / f"{model_name}.{ext}"
                file_path.write_text("saved")

        dummy_model = DummyModel()
        if not dummy_model.models_dir.exists():
            dummy_model.models_dir.mkdir(parents=True, exist_ok=True)

        file_name = "test_dummy_model_model_trainer"
        expected_path = dummy_model.models_dir / f"{file_name}.txt"
        expected_path.touch()

        self.trainer.save_model(dummy_model, file_name, ext="txt")
        assert expected_path.exists(), "Expected model file to exist after save."
        assert expected_path.read_text() == "saved", "Expected model file content to be updated by save()."


ModelTrainerTest().test_get_best_params_without_training()
ModelTrainerTest().test_optuna_train()
ModelTrainerTest().test_save_results()
ModelTrainerTest().test_save_model()
print_green("All ModelTrainer tests passed successfully!")
