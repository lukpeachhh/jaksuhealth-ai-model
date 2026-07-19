"""Evaluation routines for trained OCT segmentation models."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import MetricReport, SegmentationConfusionMatrix
from .models import count_trainable_parameters
from .visualization import plot_confusion_matrix, plot_prediction_grid


def predict_logits(
    model: nn.Module,
    images: torch.Tensor,
    use_tta: bool = False,
) -> torch.Tensor:
    """Predict logits, optionally averaging a horizontal-flip TTA pass."""

    logits = model(images)
    if not use_tta:
        return logits
    flipped_images = torch.flip(images, dims=[3])
    flipped_logits = model(flipped_images)
    restored_logits = torch.flip(flipped_logits, dims=[3])
    return 0.5 * (logits + restored_logits)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: str | Path,
    model_name: str,
    num_classes: int,
    use_tta: bool = True,
    save_samples: int = 8,
) -> dict[str, Any]:
    """Evaluate a model and export metrics and sample predictions."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.eval()
    confusion = SegmentationConfusionMatrix(num_classes=num_classes)
    samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    total_seconds = 0.0
    total_images = 0

    with torch.inference_mode():
        for images, masks in tqdm(loader, desc=f"Evaluating {model_name}"):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            logits = predict_logits(model, images, use_tta=use_tta)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_seconds += time.perf_counter() - start
            total_images += images.shape[0]

            predictions = torch.argmax(logits, dim=1)
            confusion.update(predictions, masks)

            remaining = save_samples - len(samples)
            if remaining > 0:
                count = min(remaining, images.shape[0])
                for index in range(count):
                    samples.append(
                        (
                            images[index, 0].detach().cpu().numpy(),
                            masks[index].detach().cpu().numpy(),
                            predictions[index].detach().cpu().numpy(),
                        )
                    )

    report: MetricReport = confusion.compute()
    report.per_class.to_csv(output_path / "per_class_metrics.csv", index=False)
    np.save(output_path / "confusion_matrix.npy", report.confusion_matrix)

    summary: dict[str, Any] = {
        "model_name": model_name,
        **report.overall,
        "trainable_parameters": count_trainable_parameters(model),
        "inference_seconds_per_image": total_seconds / max(total_images, 1),
        "tta_enabled": use_tta,
    }
    (output_path / "overall_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    plot_confusion_matrix(
        report.confusion_matrix,
        output_path / "confusion_matrix.png",
    )
    if samples:
        plot_prediction_grid(samples, output_path / "prediction_examples.png")

    return {
        "summary": summary,
        "per_class": report.per_class,
        "confusion_matrix": report.confusion_matrix,
    }
