import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score


def objective(trial, X_train, y_train, X_dev, y_dev, max_iter=120):
    C = trial.suggest_float("C", 0.01, 10.0, log=True)
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
    penalty = trial.suggest_categorical("penalty", ["l2", None])

    model = LogisticRegression(penalty=penalty, C=C, class_weight=class_weight, max_iter=max_iter)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_dev)
    recall = recall_score(y_dev, y_pred, pos_label="bonafide")
    return recall


class LogisticRegressionTrainer:
    def __init__(self, X_train, y_train, X_dev, y_dev):
        self.study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler())
        self.X_train = X_train
        self.y_train = y_train
        self.X_dev = X_dev
        self.y_dev = y_dev

    def train(self, n_trials=20, max_iter=120):
        self.study.optimize(
            lambda trial: objective(trial, self.X_train, self.y_train, self.X_dev, self.y_dev, max_iter=max_iter),
            n_trials=n_trials,
        )

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

    def evaluate(self, X_dev, y_dev):
        best_model = self.get_best_model(X_dev, y_dev)
        y_pred = best_model.predict(X_dev)
        recall = recall_score(y_dev, y_pred, pos_label="bonafide")
        return recall
