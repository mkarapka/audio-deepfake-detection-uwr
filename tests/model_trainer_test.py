import numpy as np

from src.common.utils import print_green
from src.models.model_trainer import ModelTrainer
from src.models.objectives import Objective


class ModelTrainerTest:
    def __init__(self):
        self.trainer = ModelTrainer(show_progress_bar=False)

    def test_optuna_train(self):
        class DummyModel:
            pass

        class DummyObjective(Objective):
            def __init__(self):
                super().__init__(classifier=DummyModel, direction="maximize")

            def __call__(self, *, trial, **params):
                return trial.suggest_float("x", 0, 1)

        X_train = np.array([[0.0], [1.0]])
        y_train = np.array([0, 1])
        X_dev = np.array([[0.0], [1.0]])
        y_dev = np.array([0, 1])

        objective = DummyObjective()

        best_params, best_value = self.trainer.optuna_train(
            objective=objective, n_trials=10, X_train=X_train, y_train=y_train, X_val=X_dev, y_val=y_dev
        )
        assert "x" in best_params, "Expected 'x' in best parameters."
        assert best_value is not None, "Expected best value to be set after training."


ModelTrainerTest().test_optuna_train()
print_green("All ModelTrainer tests passed successfully!")
