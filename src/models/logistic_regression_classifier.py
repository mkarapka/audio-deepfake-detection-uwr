import torch
import torch.nn as nn

from src.models.torch_model import TorchModel


class LogisticRegressionClassifier(TorchModel):
    def __init__(self, input_size: int, device: str = None):
        model = self._create_model(in_features=input_size)
        super().__init__(model=model, class_name=self.__class__.__name__, device=device)

    def _create_model(self, in_features: int):
        return nn.Sequential(nn.Linear(in_features=in_features, out_features=1))

    def forward(self, X: torch.Tensor):
        y_hat = self.model(X)
        return y_hat

    def save(self, file_path: str):
        payload = {
            "state_dict": self.model.state_dict(),
            "in_features": self.model[0].in_features,
        }
        torch.save(payload, file_path)

    @classmethod
    def from_pretrained(cls, file_path: str, device: str = None):
        payload = torch.load(file_path, map_location="cpu")
        in_features = payload["in_features"]
        model = cls(input_size=in_features, device=device)
        model.model.load_state_dict(payload["state_dict"])
        return model
