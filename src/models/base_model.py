from abc import ABC, abstractmethod

import numpy as np

from src.common.basic_functions import get_device
from src.common.logger import setup_logger

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


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

        if self._is_cupy_available():
            self.logger.info("CuPy is available for GPU acceleration")
        else:
            self.logger.info("CuPy is not available, using NumPy (CPU)")

    def _is_cupy_available(self):
        return CUPY_AVAILABLE

    def _to_cupy(self, np_data: np.ndarray):
        if self._is_cupy_available():
            return cp.asarray(np_data)
        self.logger.warning("CuPy is not available. Returning original NumPy array.")
        return np_data

    def _to_numpy(self, data):
        return cp.asnumpy(data)

    def _is_cupy_array(self, data):
        if self._is_cupy_available():
            return isinstance(data, cp.ndarray)
        return False

    def get_best_value(self):
        if self.study is not None:
            return self.study.best_value
        else:
            self.logger.warning("Study is not initialized. No best value available.")
            return None

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

    @abstractmethod
    def fit(self, X_train, y_train):
        """Fit eval model to training data

        Args:
            X_train (np.ndarray | cp.asarray | torch.Tensor): Training data features
            y_train (np.ndarray | cp.asarray | torch.Tensor): Training data labels
        """

    @abstractmethod
    def predict(self, X_test):
        """Predict using eval model
        Args:
            X_test (np.ndarray | cp.asarray | torch.Tensor): Test data features
        Returns:
            np.ndarray | cp.asarray | torch.Tensor: Predictions
        """
