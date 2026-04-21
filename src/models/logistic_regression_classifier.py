import torch
import torch.nn as nn

from src.models.torch_model import TorchModel


class LogisticRegressionClassifier(TorchModel):
    def __init__(self, in_features: int, device: str = None):
        model = self._create_model(in_features=in_features)
        super().__init__(model=model, class_name=self.__class__.__name__, device=device)

    def _create_model(self, in_features: int):
        return nn.Sequential(nn.Linear(in_features=in_features, out_features=1))

    def forward(self, X: torch.Tensor):
        y_hat = self.model(X)
        return y_hat
