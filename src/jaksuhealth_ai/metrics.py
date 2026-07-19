"""Confusion-matrix based segmentation metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .constants import CLASS_NAMES, NUM_CLASSES


@dataclass
class MetricReport:
    overall: dict[str, float | int]
    per_class: pd.DataFrame
    confusion_matrix: np.ndarray


class SegmentationConfusionMatrix:
    """Accumulate a multi-class confusion matrix without storing predictions."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        ignore_index: int = 255,
    ) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.matrix = torch.zeros(
            (num_classes, num_classes), dtype=torch.int64
        )

    def reset(self) -> None:
        self.matrix.zero_()

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        predictions = predictions.detach().reshape(-1).to("cpu", dtype=torch.int64)
        targets = targets.detach().reshape(-1).to("cpu", dtype=torch.int64)
        valid = (
            (targets != self.ignore_index)
            & (targets >= 0)
            & (targets < self.num_classes)
        )
        predictions = predictions[valid]
        targets = targets[valid]
        encoded = targets * self.num_classes + predictions
        counts = torch.bincount(
            encoded, minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        self.matrix += counts

    def compute(self, class_names: tuple[str, ...] = CLASS_NAMES) -> MetricReport:
        matrix = self.matrix.numpy().astype(np.float64)
        total = matrix.sum()
        epsilon = 1e-7
        rows: list[dict[str, float | int | str]] = []

        for class_id in range(self.num_classes):
            true_positive = matrix[class_id, class_id]
            false_positive = matrix[:, class_id].sum() - true_positive
            false_negative = matrix[class_id, :].sum() - true_positive
            true_negative = total - true_positive - false_positive - false_negative

            precision = true_positive / (true_positive + false_positive + epsilon)
            recall = true_positive / (true_positive + false_negative + epsilon)
            dice = 2.0 * true_positive / (
                2.0 * true_positive + false_positive + false_negative + epsilon
            )
            iou = true_positive / (
                true_positive + false_positive + false_negative + epsilon
            )
            specificity = true_negative / (
                true_negative + false_positive + epsilon
            )

            rows.append(
                {
                    "class_id": class_id,
                    "class_name": class_names[class_id],
                    "precision": float(precision),
                    "recall": float(recall),
                    "specificity": float(specificity),
                    "dice": float(dice),
                    "iou": float(iou),
                    "support_pixels": int(matrix[class_id, :].sum()),
                    "true_positive": int(true_positive),
                    "false_positive": int(false_positive),
                    "false_negative": int(false_negative),
                }
            )

        per_class = pd.DataFrame(rows)
        foreground = per_class[per_class["class_id"] != 0]

        foreground_ids = np.arange(1, self.num_classes)
        fg_tp = np.diag(matrix)[foreground_ids].sum()
        fg_fp = matrix[:, foreground_ids].sum() - fg_tp
        fg_fn = matrix[foreground_ids, :].sum() - fg_tp

        overall = {
            "pixel_accuracy": float(np.trace(matrix) / (total + epsilon)),
            "macro_precision": float(foreground["precision"].mean()),
            "macro_recall": float(foreground["recall"].mean()),
            "macro_specificity": float(foreground["specificity"].mean()),
            "macro_dice": float(foreground["dice"].mean()),
            "macro_iou": float(foreground["iou"].mean()),
            "micro_precision": float(fg_tp / (fg_tp + fg_fp + epsilon)),
            "micro_recall": float(fg_tp / (fg_tp + fg_fn + epsilon)),
            "micro_dice": float(
                2.0 * fg_tp / (2.0 * fg_tp + fg_fp + fg_fn + epsilon)
            ),
            "micro_iou": float(fg_tp / (fg_tp + fg_fp + fg_fn + epsilon)),
            "evaluated_pixels": int(total),
        }
        return MetricReport(overall, per_class, matrix.astype(np.int64))
