import optuna
import pandas as pd

from src.common.constants import Constants as consts
from src.common.logger import setup_logger
from src.models.base_model import BaseModel
from src.models.objectives import Objective


class ModelTrainer:
    def __init__(self, show_progress_bar: bool = True, garbage_collect_after_trial: bool = False):
        self.logger = setup_logger(__class__.__name__, log_to_console=True)
        self.show_progress_bar = show_progress_bar
        self.garbage_collect_after_trial = garbage_collect_after_trial
        self.study = None

    def optuna_train(self, *, objective: Objective, n_trials: int, **params):
        self.logger.info("Starting Optuna hyperparameter optimization...")
        self.study = optuna.create_study(direction=objective.direction)
        self.study.optimize(
            lambda trial: objective(trial=trial, **params),
            n_trials=n_trials,
            show_progress_bar=self.show_progress_bar,
            gc_after_trial=self.garbage_collect_after_trial,
        )

        self.logger.info(f"Best trial: {self.study.best_trial.number}")

    def get_best_params(self):
        if self.study is not None:
            return self.study.best_params
        else:
            self.logger.warning("Best parameters are not set yet.")
            return None

    def get_best_value(self):
        if self.study is not None:
            return self.study.best_value
        else:
            self.logger.warning("Best value is not set yet.")
            return None

    def save_results(self, save_file_name: str, params: dict):
        file_path = consts.train_results_dir / save_file_name
        if not consts.train_results_dir.exists():
            consts.train_results_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Saving results to {file_path}")

        save_df = pd.DataFrame(params, index=[0])
        if file_path.exists():
            existing_df = pd.read_csv(file_path)
            combined_df = pd.concat([existing_df, save_df], ignore_index=True)
            combined_df.to_csv(file_path, index=False)
        else:
            save_df.to_csv(file_path, index=False)

        self.logger.info("Results saved successfully with columns: " + ", ".join(save_df.columns))

    def save_model(self, model: BaseModel, save_file_name: str, ext: str, sub_dir: str = None):
        file_path = model._get_model_file_path(model_name=save_file_name, ext=ext, sub_dir=sub_dir)
        self.logger.info(f"Saving model to {file_path}")
        model.save(model_name=save_file_name, ext=ext, sub_dir=sub_dir)
        self.logger.info("Model saved successfully.")
