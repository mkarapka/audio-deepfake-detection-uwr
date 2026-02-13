from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.models.base_model import BaseModel


class LogisticRegressionClf(BaseModel):
    def __init__(self, **params):
        super().__init__(self.__class__.__name__)
        self.model = LogisticRegression(**params, solver="saga", n_jobs=-1)

    def get_params(self):
        return self.model.get_params()

    def fit(self, X, y):
        X = self._to_numpy(X)
        y = self._to_numpy(y)
        self.model.fit(X, y)

    def predict(self, X):
        X = self._to_numpy(X)
        return self.model.predict(X)


def objective_acc(trial, model: LogisticRegressionClf, max_iter: int, **features):
    X_train, y_train = features["X_train"], features["y_train"]
    X_dev, y_dev = features["X_dev"], features["y_dev"]

    C = trial.suggest_float("C", 0.01, 10.0, log=True)
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
    l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
    rs = 42

    clf = model(C=C, l1_ratio=l1_ratio, class_weight=class_weight, max_iter=max_iter, random_state=rs)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_dev)
    accuracy = accuracy_score(y_dev, y_pred)
    return accuracy
