"""Evaluation routines for trained OCT segmentation models."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
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
    """Predict logits with optional horizontal-flip TTA."""

    logits = model(images)

    if not use_tta:
        return logits

    flipped_images = torch.flip(images, dims=[3])
    flipped_logits = model(flipped_images)
    restored_logits = torch.flip(flipped_logits, dims=[3])

    return 0.5 * (logits + restored_logits)


def _unpack_batch(
    batch: Any,
    sample_offset: int,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Extract images, masks, and filenames from a loader batch."""

    if isinstance(batch, dict):
        images = batch["image"]
        masks = batch["mask"]
        filenames = [str(name) for name in batch["filename"]]

        return images, masks, filenames

    images, masks = batch

    filenames = [
        f"sample_{sample_offset + index:06d}.png"
        for index in range(images.shape[0])
    ]

    return images, masks, filenames


def _write_png(
    path: Path,
    array: np.ndarray,
) -> None:
    """Write a PNG file and raise an error when saving fails."""

    path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(
        str(path),
        array,
    )

    if not success:
        raise RuntimeError(f"Failed to save image: {path}")


def _save_prediction_batch(
    images: torch.Tensor,
    masks: torch.Tensor,
    predictions: torch.Tensor,
    filenames: list[str],
    images_dir: Path,
    ground_truth_dir: Path,
    predictions_dir: Path,
) -> None:
    """Save processed images, ground-truth masks, and predictions."""

    images_numpy = images.detach().cpu().numpy()
    masks_numpy = masks.detach().cpu().numpy()
    predictions_numpy = predictions.detach().cpu().numpy()

    for index, filename in enumerate(filenames):
        output_name = Path(filename).with_suffix(".png").name

        image = images_numpy[index]

        if image.ndim == 3:
            image = image[0]

        image_uint8 = np.clip(
            image * 255.0,
            0,
            255,
        ).astype(np.uint8)

        mask_uint8 = masks_numpy[index].astype(np.uint8)
        prediction_uint8 = predictions_numpy[index].astype(np.uint8)

        _write_png(
            images_dir / output_name,
            image_uint8,
        )

        _write_png(
            ground_truth_dir / output_name,
            mask_uint8,
        )

        _write_png(
            predictions_dir / output_name,
            prediction_uint8,
        )


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: str | Path,
    model_name: str,
    num_classes: int,
    use_tta: bool = True,
    save_samples: int = 8,
    save_predictions: bool = False,
) -> dict[str, Any]:
    """Evaluate a model and export metrics and predictions."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    images_dir = output_path / "images"
    ground_truth_dir = output_path / "ground_truth"
    predictions_dir = output_path / "predictions"

    if save_predictions:
        images_dir.mkdir(parents=True, exist_ok=True)
        ground_truth_dir.mkdir(parents=True, exist_ok=True)
        predictions_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    confusion = SegmentationConfusionMatrix(
        num_classes=num_classes,
    )

    samples: list[
        tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = []

    total_seconds = 0.0
    total_images = 0

    with torch.inference_mode():
        for batch in tqdm(
            loader,
            desc=f"Evaluating {model_name}",
        ):
            images, masks, filenames = _unpack_batch(
                batch,
                sample_offset=total_images,
            )

            images = images.to(
                device,
                non_blocking=True,
            )

            masks = masks.to(
                device,
                non_blocking=True,
            )

            if device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            logits = predict_logits(
                model,
                images,
                use_tta=use_tta,
            )

            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed_seconds = (
                time.perf_counter() - start_time
            )

            total_seconds += elapsed_seconds
            total_images += images.shape[0]

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            confusion.update(
                predictions,
                masks,
            )

            if save_predictions:
                _save_prediction_batch(
                    images=images,
                    masks=masks,
                    predictions=predictions,
                    filenames=filenames,
                    images_dir=images_dir,
                    ground_truth_dir=ground_truth_dir,
                    predictions_dir=predictions_dir,
                )

            remaining_samples = save_samples - len(samples)

            if remaining_samples > 0:
                sample_count = min(
                    remaining_samples,
                    images.shape[0],
                )

                images_numpy = (
                    images.detach().cpu().numpy()
                )

                masks_numpy = (
                    masks.detach().cpu().numpy()
                )

                predictions_numpy = (
                    predictions.detach().cpu().numpy()
                )

                for index in range(sample_count):
                    samples.append(
                        (
                            images_numpy[index, 0],
                            masks_numpy[index],
                            predictions_numpy[index],
                        )
                    )

    report: MetricReport = confusion.compute()

    report.per_class.to_csv(
        output_path / "per_class_metrics.csv",
        index=False,
    )

    np.save(
        output_path / "confusion_matrix.npy",
        report.confusion_matrix,
    )

    summary: dict[str, Any] = {
        "model_name": model_name,
        **report.overall,
        "trainable_parameters": count_trainable_parameters(
            model
        ),
        "inference_seconds_per_image": (
            total_seconds / max(total_images, 1)
        ),
        "tta_enabled": use_tta,
        "predictions_saved": save_predictions,
        "evaluated_images": total_images,
    }

    (
        output_path / "overall_metrics.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    plot_confusion_matrix(
        report.confusion_matrix,
        output_path / "confusion_matrix.png",
    )

    if samples:
        plot_prediction_grid(
            samples,
            output_path / "prediction_examples.png",
        )

    return {
        "summary": summary,
        "per_class": report.per_class,
        "confusion_matrix": report.confusion_matrix,
    }