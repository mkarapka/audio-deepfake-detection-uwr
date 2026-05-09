import optuna

from src.common.logger import raise_error_logger, setup_logger
from src.training.objectives import Objective


class ModelTrainer:
    def __init__(
        self,
        show_progress_bar: bool = True,
        garbage_collect_after_trial: bool = False,
        save_best_params: bool = True,
    ):
        self.study = None
        self.logger = setup_logger(__class__.__name__, log_to_console=True)

        self.show_progress_bar = show_progress_bar
        self.garbage_collect_after_trial = garbage_collect_after_trial
        self.save_best_params = save_best_params

    def optuna_train(self, *, objective: Objective, n_trials: int, **params) -> tuple[dict, float]:
        self.logger.info("Starting Optuna hyperparameter optimization...")
        self.study = optuna.create_study(direction=objective.direction)
        self.study.optimize(
            lambda trial: objective(trial=trial, **params),
            n_trials=n_trials,
            show_progress_bar=self.show_progress_bar,
            gc_after_trial=self.garbage_collect_after_trial,
        )

        if self.study is None:
            raise_error_logger(self.logger, "Study is not initialized. Cannot retrieve best parameters.")

        self.logger.info(f"Best trial: {self.study.best_trial.number}")
        self.logger.info(f"Best parameters: {self.study.best_params}")
        self.logger.info(f"Best value: {self.study.best_value}")

        return self.study.best_params, self.study.best_value
