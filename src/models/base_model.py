import numpy as np
import pandas as pd
from torch import Tensor

from src.common.basic_functions import get_device
from src.common.constants import Constants as consts
from src.common.logger import raise_error_logger, setup_logger


class BaseModel:
    def __init__(self, class_name: str, models_dir: str = consts.models_dir, include_mps=False):
        self.model = None
        self.models_dir = models_dir
        self.device = get_device(include_mps=include_mps)

        self.class_name = self.__class__.__name__ if class_name is None else class_name
        self.logger = setup_logger(f"audio_deepfake.{self.class_name}", log_to_console=True)
        self.logger.info(f"Using device: {self.device}")

    def _majority_vote(self, y_preds: np.ndarray):
        vote = int((y_preds.mean() >= 0.5))
        return vote

    def _to_numpy(self, data):
        if isinstance(data, Tensor):
            return data.cpu().numpy()
        return data

    def iterate_records(self, uq_audio_ids: pd.Series, y_preds: np.ndarray):
        unique_records_ids = uq_audio_ids.unique()

        for unique_audio_id in unique_records_ids:
            mask = uq_audio_ids == unique_audio_id
            record_predictions = y_preds[mask]

            yield record_predictions, mask

    def majority_voting(self, y_pred: np.ndarray, audio_ids: pd.Series):
        majority_voted_preds = np.full(y_pred.shape[0], -1)
        for record_preds, mask in self.iterate_records(audio_ids, y_pred):
            majority_voted_preds[mask] = self._majority_vote(record_preds)

        if np.any(majority_voted_preds == -1):
            raise_error_logger(self.logger, "Some predictions were not assigned during majority voting.")

        return majority_voted_preds

    def predict(self, X, audio_ids=None):
        if self.model is None:
            raise_error_logger(self.logger, "Model is not trained yet. Cannot perform majority voting.")

        y_pred = self.model.predict(X)
        y_pred = self._to_numpy(y_pred)

        if audio_ids is not None:
            return self.majority_voting(y_pred=y_pred, audio_ids=audio_ids)
        return y_pred
