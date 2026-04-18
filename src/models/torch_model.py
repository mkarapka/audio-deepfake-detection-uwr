import torch
import torch.nn as nn

from src.models.base_model import BaseModel


class TorchModel(BaseModel):
    def __init__(self, model: nn.Module, class_name: str, device: str = None, include_mps: bool = False):
        BaseModel.__init__(self, class_name=class_name, device=device, include_mps=include_mps)
        self.model = model

    def parameters(self):
        return self.model.parameters()

    def train_one_epoch(self, train_loader, criterion, optimizer, device):
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = self.model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_batch.size(0)
            total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total_samples += x_batch.size(0)

        return total_loss / total_samples, total_correct / total_samples

    @torch.no_grad()
    def evaluate(self, val_loader, criterion, device):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            logits = self.model(x_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * x_batch.size(0)
            total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total_samples += x_batch.size(0)

        return total_loss / total_samples, total_correct / total_samples

    def predict(self, X, audio_ids=None):
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            logits = self.model(X)
            y_pred = torch.sigmoid(logits).squeeze()

        if audio_ids is not None:
            return self.majority_voting(y_pred=y_pred.cpu().numpy(), audio_ids=audio_ids)
        return y_pred.cpu().numpy()

    def save(self, file_path):
        payload = {
            "state_dict": self.state_dict(),
            "in_features": self.model[0].in_features,
            "device": self.device,
        }
        torch.save(payload, file_path)

    def load(self, file_path):
        payload = torch.load(file_path, map_location=self.device)
        in_features = payload["in_features"]
        self.model = self._create_model(in_features=in_features)
        self.load_state_dict(payload["state_dict"])
