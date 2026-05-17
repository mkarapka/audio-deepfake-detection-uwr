import torch
import torch.nn as nn

from src.models.torch_model import TorchModel


class MlpClassifier(TorchModel):
    def __init__(self, input_size: int, hidden_sizes: list, dropout_rate: float, device: str = None):
        self.input_size = input_size
        self.hidden_sizes = list(hidden_sizes)
        self.dropout_rate = float(dropout_rate)
        model = self._create_model(input_size=input_size, hidden_sizes=hidden_sizes, dropout_rate=dropout_rate)
        super().__init__(model=model, class_name=self.__class__.__name__, device=device)

    def _create_model(self, input_size: int, hidden_sizes: list[int], dropout_rate: float):
        layers = []
        current_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            current_size = hidden_size

        layers.append(nn.Linear(current_size, 1))
        return nn.Sequential(*layers)

    def forward(self, X: torch.Tensor):
        y_hat = self.model(X)
        return y_hat

    def save(self, file_path: str):
        payload = {
            "state_dict": self.model.state_dict(),
            "input_size": self.input_size,
            "hidden_sizes": self.hidden_sizes,
            "dropout_rate": self.dropout_rate,
        }
        torch.save(payload, file_path)

    def load(self, file_path: str):
        payload = torch.load(file_path, map_location=self.device)
        self.input_size = int(payload["input_size"])
        self.hidden_sizes = list(payload["hidden_sizes"])
        self.dropout_rate = float(payload["dropout_rate"])
        self.model = self._create_model(
            input_size=self.input_size,
            hidden_sizes=self.hidden_sizes,
            dropout_rate=self.dropout_rate,
        )
        self.model.load_state_dict(payload["state_dict"])
        self.model.to(self.device)
