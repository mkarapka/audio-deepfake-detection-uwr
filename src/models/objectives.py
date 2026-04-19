from abc import ABC, abstractmethod

from optuna import Trial
from sklearn.metrics import accuracy_score
from torch.types import Tensor

from src.models.base_model import BaseModel
from src.models.logistic_regression_classifier import LogisticRegressionClassifier


class Objective(ABC):
    def __init__(self, model: BaseModel, direction: str = "maximize"):
        self.model = model
        self.direction = direction

    @abstractmethod
    def __call__(self, *args, **kwds):
        pass


class LogisticRegressionObjective(Objective):
    def __init__(self, *, X_train: Tensor, y_train: Tensor, X_val: Tensor, y_val: Tensor):
        super().__init__(model=LogisticRegressionClassifier)
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def __call__(self, *, trial: Trial, max_iter: int, metrics=accuracy_score):
        pass


class MlpObjective(Objective):
    def __init__(self):
        self.model = None

    def __call__(self, *args, **kwds):
        pass


class XGBoostObjective(Objective):
    def __init__(self):
        self.model = None

    def __call__(self, *args, **kwds):
        pass
