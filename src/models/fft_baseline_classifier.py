import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score

from src.common.logger import raise_error_logger
from src.common.record_iterator import RecordIterator
from src.models.base_model import BaseModel

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class FFTBaselineClassifier(BaseModel):
    def __init__(self, is_chunk_prediction: bool, dev_uq_audio_ids: str | None = None):
        super().__init__(model_name=self.__class__.__name__)
        self.is_chunk_prediction = is_chunk_prediction
        self.dev_uq_audio_ids = dev_uq_audio_ids

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

    def _majority_vote(self, predictions: np.ndarray) -> list[int]:
        vote = int((predictions.mean() >= 0.5))
        return [vote] * len(predictions)

    def _log_classification_report(self, y_true, y_pred, n_digits=4):
        if self._is_cupy_array(y_true):
            y_true = self._to_numpy(y_true)
        if self._is_cupy_array(y_pred):
            y_pred = self._to_numpy(y_pred)

        report = classification_report(y_true, y_pred, digits=n_digits)
        self.logger.info(f"Classification Report:\n{report}")

    def get_model(self, params):
        params["device"] = self.device
        return xgb.XGBClassifier(**params)

    def objective(self, trial, X_train, y_train, X_dev, y_dev):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 10.0, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1e-8, 25, log=True),
            "tree_method": "hist",
            "device": self.device,
        }

        self.set_model(params)
        self.fit(X_train, y_train)

        y_pred = self.predict(X_dev)
        if self._is_cupy_array(y_dev):
            y_dev = self._to_numpy(y_dev)
        if self._is_cupy_array(y_pred):
            y_pred = self._to_numpy(y_pred)

        return f1_score(y_true=y_dev, y_pred=y_pred)

    def optuna_fit(self, n_trials, X_train, y_train, X_dev, y_dev):
        y_train = self._convert_labels_to_ints(y_train, label="bonafide")
        y_dev = self._convert_labels_to_ints(y_dev, label="bonafide")
        if self._is_cupy_available():
            X_train = self._to_cupy(X_train)
            y_train = self._to_cupy(y_train)
            X_dev = self._to_cupy(X_dev)
            y_dev = self._to_cupy(y_dev)

        self.study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler())
        self.study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, X_dev, y_dev),
            n_trials=n_trials,
            gc_after_trial=True,
        )

        self.best_params = self.study.best_params

        best_model = self.get_model(self.best_params)
        best_model.fit(X_train, y_train)
        self.eval_model = best_model

        self._log_classification_report(y_true=y_dev, y_pred=self.predict(X_dev))

    def get_best_params(self):
        if self.best_params is not None:
            return self.best_params
        else:
            self.logger.warning("Best parameters are not set yet.")
            return None

    def get_best_model(self):
        if self.eval_model is not None:
            return self.eval_model
        else:
            raise_error_logger(
                self.logger,
                "Model is not trained yet. Call optuna_fit() before get_best_model().",
            )

    def set_model(self, params):
        self.eval_model = self.get_model(params)

    def fit(self, X_train, y_train, pos_label: str | None = None):
        if self.eval_model is None:
            raise_error_logger(self.logger, "Model is not set. Call set_model() before fit().")
        if pos_label is not None:
            y_train = self._convert_labels_to_ints(y_train, label=pos_label)

        if self._is_cupy_available() and not self._is_cupy_array(X_train) and not self._is_cupy_array(y_train):
            X_train = self._to_cupy(X_train)
            y_train = self._to_cupy(y_train)

        self.eval_model.fit(X_train, y_train)

    def predict(self, X):
        if self.eval_model is None:
            raise_error_logger(self.logger, "Model is not trained. Call fit() before predict().")

        y_pred = self.eval_model.predict(X)
        if self.is_chunk_prediction and self.dev_uq_audio_ids is not None:
            predictions = self.majority_voting(X, self.dev_uq_audio_ids)
        else:
            predictions = y_pred

        return predictions
