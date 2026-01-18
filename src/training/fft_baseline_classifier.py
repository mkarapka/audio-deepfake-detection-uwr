import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score

from src.common.basic_functions import get_device, setup_logger
from src.training.record_iterator import RecordIterator


class FFTBaselineClassifier:
    def __init__(self, X_train, y_train, X_dev, meta_dev):
        self.logger = setup_logger(self.__class__.__name__, log_to_console=True)
        self.X_train = X_train
        self.y_train = (y_train == "bonafide").astype(int)
        self.X_dev = X_dev
        self.y_dev = (meta_dev["target"] == "bonafide").astype(int)
        self.meta_dev = meta_dev

    def _predict_all_records(self, model):
        def majority_vote(predictions: np.ndarray) -> list[int]:
            vote = int((predictions.mean() >= 0.5))
            return [vote] * len(predictions)

        record_iterator = RecordIterator()
        results = np.full(self.X_dev.shape[0], -1)

        predictions = model.predict(self.X_dev)
        for record_preds, mask in record_iterator.iterate_records(self.meta_dev, predictions):
            majority_voted_preds = majority_vote(record_preds)
            results[mask] = majority_voted_preds

        if np.any(results == -1):
            self.logger.error("Some records were not predicted!")
        return results

    def objective_f1(self, trial, max_iter, is_partial):
        n_neg = np.sum(self.y_train == 0)
        n_pos = np.sum(self.y_train == 1)
        scale_pos_weight = n_neg / n_pos
        self.logger.info(f"Scale pos weight: {scale_pos_weight}")
        self.logger.info(f"Number of positive samples: {n_pos}, Number of negative samples: {n_neg}")

        device = get_device(include_mps=False)
        scale_tree_method = "hist"
        self.logger.info(f"Using tree method: {scale_tree_method}")
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
            "tree_method": scale_tree_method,
            "device": device,
        }

        model = xgb.XGBClassifier(**params)
        model.fit(self.X_train, self.y_train)
        y_pred = self._predict_all_records(model) if is_partial else model.predict(self.X_dev)
        f1 = f1_score(self.y_dev, y_pred)

        return f1

    def train(self, n_trials, max_iter, is_partial):
        def objective_with_data(trial):
            return self.objective_f1(
                trial,
                max_iter=max_iter,
                is_partial=is_partial,
            )

        self.study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler())
        self.study.optimize(objective_with_data, n_trials=n_trials, gc_after_trial=True)
        self.logger.info(
            f"Classification report on dev set:\n{
                classification_report(
                    self.y_dev,
                    self._predict_all_records(
                        self.get_best_model()))}"
        )

    def get_best_value(self):
        return self.study.best_value

    def get_best_params(self):
        return self.study.best_params
