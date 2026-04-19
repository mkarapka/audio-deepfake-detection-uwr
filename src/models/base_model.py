from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from torch import Tensor

from src.common.basic_functions import get_device
from src.common.constants import Constants as consts
from src.common.logger import raise_error_logger, setup_logger


class BaseModel(ABC):
    def __init__(self, class_name: str, models_dir: Path = consts.models_dir, device: str = None, include_mps=False):
        self.model = None
        self.models_dir = models_dir
        self.device = get_device(include_mps=include_mps) if device is None else device

        self.class_name = self.__class__.__name__ if class_name is None else class_name
        self.logger = setup_logger(f"audio_deepfake.{self.class_name}", log_to_console=True)
        self.logger.info(f"Using device: {self.device}")

    def _get_model_file_path(self, model_name: str, ext: str, sub_dir: str = None) -> Path:
        if not self.models_dir.exists():
            self.models_dir.mkdir(parents=True, exist_ok=True)
        if sub_dir is not None:
            sub_dir_path = self.models_dir / sub_dir
            if not sub_dir_path.exists():
                sub_dir_path.mkdir(parents=True, exist_ok=True)

        file_name = f"{model_name}.{ext}"
        model_dir_path = self.models_dir / sub_dir if sub_dir else self.models_dir
        model_file_path = model_dir_path / file_name
        if not model_file_path.exists():
            raise_error_logger(self.logger, f"Model file not found: {model_file_path}")
        return model_file_path

    def _to_numpy(self, data):
        if isinstance(data, Tensor):
            return data.cpu().numpy()
        return data

    def majority_voting(self, y_pred: np.ndarray, audio_ids: pd.Series):
        df = pd.DataFrame({"audio_id": audio_ids, "pred": y_pred})
        mean_preds = df.groupby("audio_id")["pred"].transform("mean")
        majority_voted_preds = (mean_preds >= 0.5).astype(int).values
        return majority_voted_preds

    @abstractmethod
    def predict(self, X, audio_ids=None):
        pass

    @abstractmethod
    def load(self, file_path: str):
        pass

    @abstractmethod
    def save(self, file_path: str):
        pass
