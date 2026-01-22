from abc import ABC, abstractmethod

from src.common.basic_functions import get_device
from src.common.logger import setup_logger
import numpy as np
from src.common.record_iterator import RecordIterator
from src.common.logger import raise_error_logger
class BaseModel(ABC):
    def __init__(self, model_name, include_mps=False):
        self.model_name = model_name
        self.eval_model = None
        self.study = None
        self.best_params = None
        self.device = get_device(include_mps=include_mps)

        self.logger = setup_logger(f"audio_deepfake.{self.model_name}", log_to_console=True)
        self.logger.info(f"Initialized model: {self.model_name}")
        self.logger.info(f"Using device: {self.device}")

    def _convert_labels_to_ints(self, y, label: str):
        return (y == label).astype(int)

    def get_best_value(self):
        if self.study is not None:
            return self.study.best_value
        else:
            self.logger.warning("Study is not initialized. No best value available.")
            return None

    def majority_voting(self, X_dev, dev_uq_audio_ids):
        if self.eval_model is None:
            raise_error_logger(self.logger, "Model is not trained. Call fit() before majority_voting().")

        y_pred = self.eval_model.predict(X_dev)
        record_iterator = RecordIterator()
        predictions = np.full(X_dev.shape[0], -1)
        for record_preds, mask in record_iterator.iterate_records(dev_uq_audio_ids, y_pred):
            majority_voted_preds = self._majority_vote(record_preds)
            predictions[mask] = majority_voted_preds

        if np.any(predictions == -1):
            raise_error_logger(self.logger, "Some records were not predicted!")

        return predictions

    @abstractmethod
    def objective(self, trial, params):
        """Create optuna objective function

        Args:
            trial (any): Optuna trial object
            params (any): Parameters for the objective function
        """

    @abstractmethod
    def optuna_fit(self, n_trials: int, X_train, y_train):
        """Optimize model using Optuna
        Args:
            n_trials (int): Number of Optuna trials
            X_train (np.ndarray | cp.asarray | torch.Tensor): Training data features
            y_train (np.ndarray | cp.asarray | torch.Tensor): Training data labels
        """

    @abstractmethod
    def get_model(self, params):
        """Get evaluation model with given parameters
        Args:
            params (dict): Model parameters
        Returns:
            any: Evaluation model
        """

    @abstractmethod
    def get_best_params(self):
        """Get best parameters found by Optuna
        Returns:
            dict: Best parameters
        """

    # @abstractmethod
    # def fit(self, X_train, y_train):
    #     """Fit eval model to training data

    #     Args:
    #         X_train (np.ndarray | cp.asarray | torch.Tensor): Training data features
    #         y_train (np.ndarray | cp.asarray | torch.Tensor): Training data labels
    #     """

    @abstractmethod
    def predict(self, X_test):
        """Predict using eval model
        Args:
            X_test (np.ndarray | cp.asarray | torch.Tensor): Test data features
        Returns:
            np.ndarray | cp.asarray | torch.Tensor: Predictions
        """
