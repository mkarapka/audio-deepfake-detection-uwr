import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.functional.classification import binary_eer

from src.evaluation.binary_evaluator import BinaryEvaluator


class DummyBinaryModel:
    """Minimal model stub compatible with BinaryEvaluator.evaluate()."""

    def __init__(self):
        self.device = torch.device("cpu")
        self.linear = nn.Linear(1, 1)
        with torch.no_grad():
            # Make predictions correlate with X for stable metrics.
            self.linear.weight.fill_(4.0)
            self.linear.bias.fill_(-2.0)

    @torch.no_grad()
    def evaluate(self, *, val_loader, criterion, device, threshold: float = 0.5):
        self.linear.to(device)
        self.linear.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_probs = []
        all_labels = []

        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = self.linear(x_batch)
            loss = criterion(logits, y_batch)

            probs = torch.sigmoid(logits)
            y_pred = (probs >= threshold).float()

            total_loss += loss.item() * x_batch.size(0)
            total_correct += (y_pred == y_batch).sum().item()
            total_samples += x_batch.size(0)

            all_probs.append(probs.cpu())
            all_labels.append(y_batch.cpu())

        y_true = torch.cat(all_labels).int()
        y_probs = torch.cat(all_probs)

        return total_loss / total_samples, total_correct / total_samples, y_true, y_probs


class BinaryEvaluatorTest:
    def test_compute_metrics_keys_and_ranges(self):
        evaluator = BinaryEvaluator()

        y_true = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int)
        y_probs = torch.tensor([0.05, 0.1, 0.2, 0.35, 0.6, 0.75, 0.9, 0.95], dtype=torch.float)

        metrics = evaluator._compute_metrics(y_true=y_true, y_probs=y_probs)

        # EER should match torchmetrics implementation (within tolerance)
        eer_tm = binary_eer(y_probs, y_true)
        # torchmetrics may return (eer, threshold) or just eer depending on version
        if isinstance(eer_tm, (tuple, list)):
            eer_tm = eer_tm[0]
        eer_tm = float(torch.as_tensor(eer_tm).item())
        assert metrics["eer"] == pytest.approx(eer_tm, abs=1e-6)
        print(f"Computed EER: {metrics['eer']:.4f}, TorchMetrics EER: {eer_tm:.4f}")

        expected_keys = {
            "eer",
            "accuracy_eer_threshold",
            "precision_eer_threshold",
            "recall_eer_threshold",
            "f1_score_eer_threshold",
            "accuracy_base_threshold",
            "precision_base_threshold",
            "recall_base_threshold",
            "f1_score_base_threshold",
            "auroc",
        }
        assert set(metrics.keys()) == expected_keys
        for key, value in metrics.items():
            assert 0.0 <= value <= 1.0, f"{key} out of range: {value}"

    def test_evaluate_returns_prefixed_metrics(self):
        torch.manual_seed(27)

        # 0..1 values; label is 1 if x > 0.5 else 0
        X = torch.linspace(0, 1, steps=64).unsqueeze(1)
        y = (X > 0.5).float()

        loader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=False)
        model = DummyBinaryModel()
        evaluator = BinaryEvaluator(criterion=nn.BCEWithLogitsLoss())

        out = evaluator.evaluate(model=model, dataloader=loader, log_prefix="test")

        # Basic presence
        assert "test/loss" in out
        assert "test/eer" in out
        assert "test/auroc" in out
        assert "test/accuracy" in out
        assert "test/f1" in out

        # Prefixing + ranges
        for key, value in out.items():
            assert key.startswith("test/"), f"Key not prefixed: {key}"
            assert isinstance(value, float), f"Expected float for {key}, got {type(value)}"
            # loss can be > 1, others should be within [0, 1]
            if not key.endswith("/loss"):
                assert 0.0 <= value <= 1.0, f"{key} out of range: {value}"


def test_binary_evaluator_compute_metrics_keys_and_ranges():
    BinaryEvaluatorTest().test_compute_metrics_keys_and_ranges()


def test_binary_evaluator_evaluate_returns_prefixed_metrics():
    BinaryEvaluatorTest().test_evaluate_returns_prefixed_metrics()
