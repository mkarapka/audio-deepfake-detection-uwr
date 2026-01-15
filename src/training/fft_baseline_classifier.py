import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import f1_score

from src.common.basic_functions import setup_logger, get_device


class FFTBaselineClassifier:
    def __init__(self, X_train, y_train, X_dev, y_dev):
        self.logger = setup_logger(self.__class__.__name__, log_to_console=True)
        self.X_train = X_train
        self.y_train = (y_train == "bonafide").astype(int)
        self.X_dev = X_dev
        self.y_dev = (y_dev == "bonafide").astype(int)

    def objective_f1(self, trial, X_train, y_train, X_dev, y_dev, max_iter=120):
        n_neg = np.sum(y_train == 0)
        n_pos = np.sum(y_train == 1)
        scale_pos_weight = n_neg / n_pos
        self.logger.info(f"Scale pos weight: {scale_pos_weight}")
        self.logger.info(f"Number of positive samples: {n_pos}, Number of negative samples: {n_neg}")

        device = get_device()
        self.logger.info(f"Using device: {device}")

        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 10.0, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "scale_pos_weight": scale_pos_weight,
            "tree_method": "hist",
            "device": device,
        }

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_dev)
        f1 = f1_score(y_dev, y_pred)

        return f1

    def train(self, n_trials=20, max_iter=120):
        def objective_with_data(trial):
            return self.objective_f1(
                trial,
                X_train=self.X_train,
                y_train=self.y_train,
                X_dev=self.X_dev,
                y_dev=self.y_dev,
                max_iter=max_iter,
            )

        self.study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler())
        self.study.optimize(objective_with_data, n_trials=n_trials, gc_after_trial=True)

    def get_best_value(self):
        return self.study.best_value

    def get_best_params(self):
        return self.study.best_params
