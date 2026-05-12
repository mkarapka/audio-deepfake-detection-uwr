import torch
import torch.nn as nn

from src.models.torch_model import TorchModel


class MlpClassifier(TorchModel):
    def __init__(self, input_size: int, hidden_sizes: list, dropout_rate: float, device: str = None):
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
