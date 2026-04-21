from abc import abstractmethod

import torch
import torch.nn as nn

from src.models.base_model import BaseModel


class TorchModel(BaseModel):
    def __init__(self, model: nn.Module, class_name: str, device: str = None, include_mps: bool = True):
        super().__init__(class_name=class_name, device=device, include_mps=include_mps)
        self.model = model.to(self.device)

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
        all_preds = []
        all_labels = []

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        self.model.eval()
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            logits = self.model(x_batch)
            loss = criterion(logits, y_batch)
            y_pred = logits.argmax(dim=1)

            total_loss += loss.item() * x_batch.size(0)
            total_correct += (y_pred == y_batch).sum().item()
            total_samples += x_batch.size(0)

            all_preds.append(y_pred.detach().cpu())
            all_labels.append(y_batch.detach().cpu())

        val_loss = total_loss / total_samples
        val_acc = total_correct / total_samples

        y_true = torch.cat(all_labels)
        y_pred = torch.cat(all_preds)

        return val_loss, val_acc, y_true, y_pred

    def predict(self, X, audio_ids=None):
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            logits = self.model(X)
            y_pred = torch.sigmoid(logits).squeeze()
            y_pred = y_pred.detach().cpu()

        if audio_ids:
            return self.majority_voting(y_pred=y_pred, audio_ids=audio_ids)
        return y_pred

    def save(self, file_path):
        payload = {
            "state_dict": self.state_dict(),
            "in_features": self.model[0].in_features,
        }
        torch.save(payload, file_path)

    def load(self, file_path):
        payload = torch.load(file_path, map_location=self.device)
        in_features = payload["in_features"]
        self.model = self._create_model(in_features=in_features)
        self.load_state_dict(payload["state_dict"])
        self.model.to(self.device)

    @abstractmethod
    def _create_model(self, in_features):
        pass
