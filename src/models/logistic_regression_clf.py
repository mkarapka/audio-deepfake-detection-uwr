import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

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

    def save(self, file_path):
        joblib.dump(self.model, file_path)

    def load(self, file_path):
        self.model = joblib.load(file_path)


def objective_acc(
    trial, model: LogisticRegressionClf, X_train, y_train, X_dev, y_dev, max_iter=None, metrics=accuracy_score
):
    C = trial.suggest_float("C", 0.01, 10.0, log=True)
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
    l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
    penality = trial.suggest_categorical("penalty", [None, "elasticnet"])
    if max_iter is None:
        max_iter = trial.suggest_int("max_iter", 200, 500)
    rs = 42

    clf = model(
        C=C,
        l1_ratio=l1_ratio,
        penalty=penality,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=rs,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_dev)
    score = metrics(y_dev, y_pred)
    return score
