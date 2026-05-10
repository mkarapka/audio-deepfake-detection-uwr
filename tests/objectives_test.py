import numpy as np
import optuna
import pandas as pd
import pytest
from torch.utils.data import DataLoader

from src.datasets.audio_dataset import AudioDataset
from src.training.objectives import (
    LogisticRegressionObjective,
    MlpObjective,
    XGBoostObjective,
)


@pytest.fixture
def dummy_train_loader():
    X = np.random.randn(100, 16).astype(np.float32)
    y = pd.DataFrame({"target": np.random.choice(["bonafide", "spoof"], size=100)})
    dataset = AudioDataset(metadata=y, features=X)
    return DataLoader(dataset, batch_size=32, shuffle=True)


@pytest.fixture
def dummy_val_loader():
    X = np.random.randn(50, 16).astype(np.float32)
    y = pd.DataFrame({"target": np.random.choice(["bonafide", "spoof"], size=50)})
    dataset = AudioDataset(metadata=y, features=X)
    return DataLoader(dataset, batch_size=32, shuffle=False)


@pytest.fixture
def dummy_numpy_data():
    X_train = np.random.randn(100, 16).astype(np.float32)
    y_train = np.random.randint(0, 2, size=(100,)).astype(np.float32)
    X_dev = np.random.randn(50, 16).astype(np.float32)
    y_dev = np.random.randint(0, 2, size=(50,)).astype(np.float32)
    return X_train, y_train, X_dev, y_dev


def test_logistic_regression_objective_runs(dummy_train_loader, dummy_val_loader):
    objective = LogisticRegressionObjective(
        train_loader=dummy_train_loader,
        val_loader=dummy_val_loader,
    )

    study = optuna.create_study(direction="minimize")

    # Wykonujemy tylko 1 próbę (n_trials=1) aby upewnić się że funkcja przechodzi
    study.optimize(lambda trial: objective(trial=trial, epochs=2, use_pos_weight=True), n_trials=1)

    assert study.best_value is not None
    assert isinstance(study.best_value, float)


def test_mlp_objective_runs(dummy_train_loader, dummy_val_loader):
    objective = MlpObjective(
        train_loader=dummy_train_loader,
        val_loader=dummy_val_loader,
    )

    study = optuna.create_study(direction="minimize")

    study.optimize(lambda trial: objective(trial=trial, epochs=2, use_pos_weight=True), n_trials=1)

    assert study.best_value is not None
    assert isinstance(study.best_value, float)


def test_xgboost_objective_runs(dummy_numpy_data):
    X_train, y_train, X_dev, y_dev = dummy_numpy_data

    objective = XGBoostObjective(
        X_train=X_train,
        y_train=y_train,
        X_dev=X_dev,
        y_dev=y_dev,
    )

    study = optuna.create_study(direction="minimize")

    study.optimize(lambda trial: objective(trial=trial, early_stopping_rounds=1), n_trials=1)

    assert study.best_value is not None
    assert isinstance(study.best_value, float)
