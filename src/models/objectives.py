from abc import ABC, abstractmethod

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from optuna import Trial
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import binary_eer

from src.common.logger import setup_logger
from src.models.logistic_regression_classifier import LogisticRegressionClassifier
from src.models.torch_model import TorchModel


class Objective(ABC):
    def __init__(self, classifier: TorchModel | xgb.Booster, direction: str = "minimize"):
        self.logger = setup_logger(__class__.__name__, log_to_console=True)
        self.classifier = classifier
        self.direction = direction

    @abstractmethod
    def __call__(self, *args, **kwds):
        pass


class LogisticRegressionObjective(Objective):
    def __init__(self, *, train_loader: DataLoader, val_loader: DataLoader):
        self.train_loader = train_loader
        self.val_loader = val_loader

        features, _ = next(iter(train_loader))
        in_features = features.shape[-1]
        super().__init__(classifier=LogisticRegressionClassifier(in_features=in_features))

    def __call__(self, *, trial: Trial, epochs: int, use_pos_weight: bool, logging_percent_threshold: float = 0.1):
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
            train_loss, train_acc = self.classifier.train_one_epoch(
                train_loader=self.train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=self.classifier.device,
            )
            val_loss, val_acc, y_true, y_pred = self.classifier.evaluate(
                val_loader=self.val_loader,
                criterion=criterion,
                device=self.classifier.device,
            )

            score = binary_eer(preds=y_pred, target=y_true).item()
            trial.report(score, step=epoch)

            if epoch % max(1, epochs // int(1 / logging_percent_threshold)) == 0:
                self.logger.info(
                    f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, EER: {score:.4f}"
                )
                
            if score < best_score:
                best_score = score

            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_score


class XGBoostObjective(Objective):
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, X_dev: np.ndarray, y_dev: np.ndarray):
        super().__init__(classifier=None)
        self.dmatrix_train = xgb.DMatrix(X_train, label=y_train)
        self.dmatrix_dev = xgb.DMatrix(X_dev, label=y_dev)

    def __call__(self, *, trial: Trial, early_stopping_rounds: int):
        neg_count = np.sum(self.dmatrix_train.get_label() == 0)
        pos_count = np.sum(self.dmatrix_train.get_label() == 1)
        base_spw = neg_count / pos_count if pos_count > 0 else 1

        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5 * base_spw, 2.0 * base_spw, log=True),
            "tree_method": "hist",
            "device": self.device,
        }
        num_boost_round = trial.suggest_int("num_boost_round", 100, 1000)
        evals = [(self.dmatrix_dev, "validation")]

        self.classifier = xgb.train(
            params,
            self.dmatrix_train,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
        )
        y_pred = self.classifier.predict(self.dmatrix_dev)

        return binary_eer(preds=torch.tensor(y_pred), target=torch.tensor(self.dmatrix_dev.get_label())).item()


class MlpObjective(Objective):
    def __init__(self):
        self.model = None

    def __call__(self, *args, **kwds):
        pass
