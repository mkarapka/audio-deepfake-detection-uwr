from functools import partial

import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def objective_f1(trial, X_train, y_train, X_dev, y_dev, max_iter=120):
    C = trial.suggest_float("C", 0.01, 10.0, log=True)
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
    penalty = trial.suggest_categorical("penalty", ["l2", None])

    model = LogisticRegression(
        penalty=penalty, C=C, class_weight=class_weight, max_iter=max_iter
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_dev)
    recall = f1_score(y_dev, y_pred, pos_label="bonafide")
    return recall


def objective_acc(trial, X_train, y_train, X_dev, y_dev, max_iter=120):
    C = trial.suggest_float("C", 0.01, 10.0, log=True)
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
    penalty = trial.suggest_categorical("penalty", ["l2", None])

    model = LogisticRegression(
        penalty=penalty, C=C, class_weight=class_weight, max_iter=max_iter
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_dev)
    accuracy = accuracy_score(y_dev, y_pred)
    return accuracy


class LogisticRegressionClassifier:
    def __init__(self, X_train, y_train, X_dev, y_dev, objective="f1"):
        self.study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.RandomSampler()
        )
        self.X_train = X_train
        self.y_train = y_train
        self.X_dev = X_dev
        self.y_dev = y_dev
        self.objective = objective

    def train(self, n_trials=20, max_iter=120):
        if self.objective == "f1":
            objective_func = objective_f1
        else:
            objective_func = objective_acc

        objective_with_data = partial(
            objective_func,
            X_train=self.X_train,
            y_train=self.y_train,
            X_dev=self.X_dev,
            y_dev=self.y_dev,
            max_iter=max_iter,
        )

        self.study.optimize(objective_with_data, n_trials=n_trials, gc_after_trial=True)

    def get_best_model(self):
        best_params = self.study.best_params
        print(f"Best hyperparameters: {best_params}")
        best_model = LogisticRegression(
            penalty=best_params["penalty"],
            C=best_params["C"],
            class_weight=best_params["class_weight"],
        )
        best_model.fit(self.X_train, self.y_train)
        return best_model

    def get_best_value(self):
        return self.study.best_value

    def get_best_params(self):
        return self.study.best_params
