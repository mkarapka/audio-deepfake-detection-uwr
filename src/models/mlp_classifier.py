import optuna
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from src.common.basic_functions import get_batch_size
from src.models.base_model import BaseModel


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout_rate):
        super(MLP, self).__init__()
        layers = []
        for i in range(n_layers):
            in_size = input_size if i == 0 else hidden_size
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(1)


class MLPClassifier(BaseModel):
    def __init__(self, is_chunk_prediction=False):
        super().__init__(model_name=__class__.__name__, include_mps=True)

    def _np_to_tensor(self, np_data):
        if isinstance(np_data, torch.Tensor):
            return np_data.to(self.device)
        return torch.tensor(np_data, dtype=torch.float32).to(self.device)

    def train_model(self, model, optimizer, criterion, train_loader):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y.float())
            loss.backward()
            optimizer.step()

    def evaluate_model_no_score(self, model, dev_loader):
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_X, batch_y in dev_loader:
                logits = model(batch_X)
                preds = torch.sigmoid(logits) >= 0.5
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        return all_labels, all_preds

    def evaluate_model(self, model, dev_loader):
        all_labels, all_preds = self.evaluate_model_no_score(model, dev_loader)
        f1 = f1_score(all_labels, all_preds)
        return f1

    def objective(self, trial, X_train: torch.Tensor, y_train: torch.Tensor, X_dev: torch.Tensor, y_dev: torch.Tensor):
        params = {
            "n_layers": trial.suggest_int("n_layers", 1, 2),
            "hidden_size": trial.suggest_int("hidden_size", 64, 256, step=64),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.4),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "epochs": trial.suggest_int("epochs", 10, 25),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        }

        model = self.get_model(
            input_size=X_train.shape[1],
            hidden_size=params["hidden_size"],
            n_layers=params["n_layers"],
            dropout_rate=params["dropout_rate"],
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
        criterion = torch.nn.BCEWithLogitsLoss()

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=params["batch_size"], shuffle=True)
        dev_loader = DataLoader(TensorDataset(X_dev, y_dev), batch_size=params["batch_size"], shuffle=True)

        for epoch in range(params["epochs"]):
            self.train_model(model, optimizer, criterion, train_loader)

            f1 = self.evaluate_model(model, dev_loader)
            trial.report(f1, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return f1

    def get_model(self, input_size, hidden_size, n_layers, dropout_rate):
        return MLP(
            input_size=input_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            dropout_rate=dropout_rate
        ).to(self.device)

    def optuna_fit(self, n_trials, X_train, y_train, X_dev, y_dev):
        y_train = self._convert_labels_to_ints(y=y_train, label="bonafide")
        y_dev = self._convert_labels_to_ints(y=y_dev, label="bonafide")

        X_train_tensor = self._np_to_tensor(X_train)
        y_train_tensor = self._np_to_tensor(y_train)
        X_dev_tensor = self._np_to_tensor(X_dev)
        y_dev_tensor = self._np_to_tensor(y_dev)

        self.study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler())
        self.study.optimize(
            lambda trial: self.objective(trial, X_train_tensor, y_train_tensor, X_dev_tensor, y_dev_tensor),
            n_trials=n_trials,
            gc_after_trial=True,
        )

        self.best_params = self.study.best_params

    def save_model(self):
        pass

    def load_model(self):
        pass

    def get_best_params(self):
        return super().get_best_params()

    def predict(self, X_test):
        return super().predict(X_test)


if __name__ == "__main__":
    clf = MLPClassifier()
