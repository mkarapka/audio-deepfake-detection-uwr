from abc import ABC, abstractmethod

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from optuna import Trial
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import (
    binary_accuracy,
    binary_eer,
    binary_f1_score,
    binary_precision,
    binary_recall,
)

from src.common.logger import WandbLogger, setup_logger
from src.models.logistic_regression_classifier import LogisticRegressionClassifier
from src.models.mlp_classifier import MlpClassifier
from src.models.torch_model import TorchModel


class Objective(ABC):
    def __init__(self, class_name: str, direction: str = "maximize", wandb_run=None):
        self.logger = setup_logger(class_name, log_to_console=True)
        self.wandb_logger = WandbLogger(self.logger, run=wandb_run)
        self.direction = direction

    @abstractmethod
    def __call__(self, *args, **kwds):
        pass


class TorchBinaryObjective(Objective):
    def __init__(self, *, class_name: str, train_loader, val_loader, wandb_run=None):
        super().__init__(class_name=class_name, wandb_run=wandb_run)
        self.train_loader = train_loader
        self.val_loader = val_loader

        features, _ = next(iter(train_loader))
        self.in_features = features.shape[-1]

    @abstractmethod
    def build_classifier(self, *, trial: Trial) -> TorchModel:
        pass

    def update_score(self, prev_score: float, new_score: float):
        best_score = min(prev_score, new_score) if self.direction == "minimize" else max(prev_score, new_score)
        self.wandb_logger.log_metrics({"best_score": best_score})
        return best_score

    def suggest_optim_params(self, trial: Trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        return lr, weight_decay

    def suggest_pos_weight(self, trial: Trial, use_pos_weight: bool, device):
        if not use_pos_weight:
            return None
        pw = trial.suggest_float("pos_weight", 5.0, 40.0)
        return torch.tensor([pw], device=device)

    def __call__(
        self, *, trial: Trial, epochs: int, use_pos_weight: bool = True, logging_percent_threshold: float = 0.1
    ):
        classifier = self.build_classifier(trial=trial)

        lr, weight_decay = self.suggest_optim_params(trial)
        pos_weight = self.suggest_pos_weight(trial, use_pos_weight, classifier.device)

        optimizer = optim.AdamW(classifier.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_score = float("-inf") if self.direction == "maximize" else float("inf")
        for epoch in range(epochs):
            train_loss, train_acc = classifier.train_one_epoch(
                train_loader=self.train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=classifier.device,
            )
            val_loss, val_acc, y_true, y_probs = classifier.evaluate(
                val_loader=self.val_loader,
                criterion=criterion,
                device=classifier.device,
            )

            score = val_acc
            trial.report(score, step=epoch)

            if epoch % max(1, epochs // int(1 / logging_percent_threshold)) == 0:
                self.logger.info(f"==== Epoch {epoch + 1}/{epochs} ====")
                self.wandb_logger.log_metrics(
                    {
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "val_precision": binary_precision(y_probs, y_true).item(),
                        "val_recall": binary_recall(y_probs, y_true).item(),
                        "val_f1": binary_f1_score(y_probs, y_true).item(),
                        "val_eer": binary_eer(preds=y_probs, target=y_true).item(),
                    },
                )

            best_score = self.update_score(best_score, score)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_score


class LogisticRegressionObjective(TorchBinaryObjective):
    def __init__(self, *, train_loader: DataLoader, val_loader: DataLoader, wandb_run=None):
        super().__init__(
            class_name=__class__.__name__,
            train_loader=train_loader,
            val_loader=val_loader,
            wandb_run=wandb_run,
        )

    def build_classifier(self, *, trial: Trial):
        return LogisticRegressionClassifier(input_size=self.in_features)


class MlpObjective(TorchBinaryObjective):
    def __init__(self, *, train_loader: DataLoader, val_loader: DataLoader, wandb_run=None):
        super().__init__(
            class_name=__class__.__name__,
            train_loader=train_loader,
            val_loader=val_loader,
            wandb_run=wandb_run,
        )

    def _get_suggested_hidden_sizes(self, trial: Trial, step: int = 32) -> list[int]:
        n_layers = trial.suggest_int("n_layers", 1, 4)
        hidden_sizes = []

        high = self.in_features
        for i in range(n_layers):
            high = max(step, (high // step) * step)
            out_size = trial.suggest_int(f"hidden_size_{i}", step, high, step=step)
            hidden_sizes.append(out_size)
            high = out_size

        return hidden_sizes

    def build_classifier(self, *, trial: Trial):
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        hidden_sizes = self._get_suggested_hidden_sizes(trial)
        return MlpClassifier(input_size=self.in_features, hidden_sizes=hidden_sizes, dropout_rate=dropout_rate)


class XGBoostObjective(Objective):
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, X_dev: np.ndarray, y_dev: np.ndarray, wandb_run=None):
        super().__init__(class_name=__class__.__name__, wandb_run=wandb_run)
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
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
        num_boost_round = trial.suggest_int("num_boost_round", 100, 1000)
        evals = [(self.dmatrix_dev, "validation")]

        classifier = xgb.train(
            params,
            self.dmatrix_train,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
        )
        y_pred = classifier.predict(self.dmatrix_dev)

        return binary_accuracy(torch.tensor(y_pred), torch.tensor(self.dmatrix_dev.get_label())).item()
