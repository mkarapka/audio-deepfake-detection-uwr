import numpy as np
import optuna
import pandas as pd

from src.common.constants import Constants as consts
from src.common.logger import setup_logger
from src.models.base_model import BaseModel


class ModelTrainer:
    def __init__(self, is_majority_voting: bool = False):
        self.logger = setup_logger(__class__.__name__, log_to_console=True)
        self.is_majority_voting = is_majority_voting
        self.study = None

    def _convert_labels_to_ints(self, y: pd.Series, pos_label: str) -> np.ndarray:
        return (y == pos_label).astype(int)

    def get_target(self, metadata: pd.DataFrame, pos_label="bonafide") -> np.ndarray:
        y = self._convert_labels_to_ints(metadata["target"], pos_label=pos_label)
        return y

    def optuna_train(self, model: BaseModel, objective, n_trials: int, direct: str = "maximize", **params):
        self.logger.info("Starting Optuna hyperparameter optimization...")
        self.study = optuna.create_study(direction=direct)
        self.study.optimize(lambda trial: objective(trial, model, **params), n_trials=n_trials, show_progress_bar=True)

        self.logger.info(f"Best trial: {self.study.best_trial.number}")

    def get_best_params(self):
        if self.study is not None:
            return self.study.best_params
        else:
            self.logger.warning("Best parameters are not set yet.")
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
