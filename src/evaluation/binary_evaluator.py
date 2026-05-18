import torch
import torch.nn as nn
from torchmetrics.functional.classification import (
    binary_accuracy,
    binary_auroc,
    binary_eer,
    binary_f1_score,
    binary_precision,
    binary_recall,
    binary_roc,
)

from src.common.logger import setup_logger


class BinaryEvaluator:
    def __init__(self, criterion: nn.Module | None = None):
        self.logger = setup_logger(__class__.__name__, log_to_console=True)
        self.criterion = criterion or nn.BCEWithLogitsLoss()

    def _compute_metrics(self, *, y_true, y_probs):
        fpr, tpr, thresholds = binary_roc(y_probs, y_true)
        fnr = 1 - tpr
        eer_idx = torch.argmin(torch.abs(fpr - fnr))
        eer_threshold = thresholds[eer_idx].item()

        accuracy_eer_threshold = binary_accuracy(y_probs, y_true, threshold=eer_threshold).item()
        precision_eer_threshold = binary_precision(y_probs, y_true, threshold=eer_threshold).item()
        recall_eer_threshold = binary_recall(y_probs, y_true, threshold=eer_threshold).item()
        f1_score_eer_threshold = binary_f1_score(y_probs, y_true, threshold=eer_threshold).item()

        eer = ((fpr[eer_idx] + fnr[eer_idx]) / 2).item()
        accuracy_base_threshold = binary_accuracy(y_probs, y_true).item()
        precision_base_threshold = binary_precision(y_probs, y_true).item()
        recall_base_threshold = binary_recall(y_probs, y_true).item()
        f1_score_base_threshold = binary_f1_score(y_probs, y_true).item()
        auroc = binary_auroc(y_probs, y_true).item()

        self.logger.info(f"Computed Metrics: EER={eer:.4f}, EER from TorchMetrics={binary_eer(y_probs, y_true):.4f}")

        return [
            ("eer", eer),
            ("accuracy_eer_threshold", accuracy_eer_threshold),
            ("precision_eer_threshold", precision_eer_threshold),
            ("recall_eer_threshold", recall_eer_threshold),
            ("f1_score_eer_threshold", f1_score_eer_threshold),
            ("accuracy_base_threshold", accuracy_base_threshold),
            ("precision_base_threshold", precision_base_threshold),
            ("recall_base_threshold", recall_base_threshold),
            ("f1_score_base_threshold", f1_score_base_threshold),
            ("auroc", auroc),
        ]

    def evaluate(self, *, model, dataloader, log_prefix: str) -> dict[str, float]:
        loss, _, y_true, y_probs = model.evaluate(
            val_loader=dataloader,
            criterion=self.criterion,
            device=model.device,
        )
        self.logger.info(f"Compute Metrics on {len(y_true)} samples")
        metrics = self._compute_metrics(y_true=y_true, y_probs=y_probs)
        metrics_lst = [("loss", loss)] + metrics

        return dict((f"{log_prefix}/{name}", value) for name, value in metrics_lst)
