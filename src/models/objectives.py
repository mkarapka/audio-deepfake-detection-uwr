from abc import ABC, abstractmethod

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from optuna import Trial
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import binary_eer

from src.models.base_model import BaseModel
from src.models.logistic_regression_classifier import LogisticRegressionClassifier
from src.models.torch_model import TorchModel


class Objective(ABC):
    def __init__(self, classifier: BaseModel | TorchModel, direction: str = "maximize"):
        self.classifier = classifier
        self.direction = direction

    @abstractmethod
    def __call__(self, *args, **kwds):
        pass


class LogisticRegressionObjective(Objective):
    def __init__(self, *, in_features: int, train_loader: DataLoader, val_loader: DataLoader):
        self.train_loader = train_loader
        self.val_loader = val_loader
        super().__init__(classifier=LogisticRegressionClassifier(in_features=in_features))

    def __call__(self, *, trial: Trial, epochs: int, use_pos_weight: bool):
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        if use_pos_weight:
            pw = trial.suggest_float("pos_weight", 5.0, 40.0)
            pos_weight = torch.tensor([pw], device=self.classifier.device)
        else:
            pos_weight = None

        optimizer = optim.AdamW(self.classifier.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        best_score = float("inf")

        for epoch in range(epochs):
            _, _ = self.classifier.train_one_epoch(
                train_loader=self.train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=self.classifier.device,
            )
            _, _, y_true, y_pred = self.classifier.evaluate(
                val_loader=self.val_loader,
                criterion=criterion,
                device=self.classifier.device,
            )

            score = binary_eer(preds=y_pred, target=y_true).item()
            trial.report(score, step=epoch)

            if score < best_score:
                best_score = score

            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_score


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
