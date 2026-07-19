"""Visualization helpers for segmentation outputs and experiment reports."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .constants import CLASS_COLORS, CLASS_NAMES


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert integer labels to an RGB image."""

    palette = np.asarray(CLASS_COLORS, dtype=np.uint8)
    safe_mask = np.clip(mask.astype(np.int64), 0, len(palette) - 1)
    return palette[safe_mask]


def apply_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend lesion colors onto a grayscale or RGB OCT image."""

    if image.ndim == 2:
        image_rgb = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image.astype(np.uint8).copy()
    colored = colorize_mask(mask)
    overlay = image_rgb.copy()
    foreground = mask > 0
    overlay[foreground] = (
        image_rgb[foreground] * (1.0 - alpha) + colored[foreground] * alpha
    ).astype(np.uint8)
    return overlay


def plot_training_curves(history_csv: str | Path, output_path: str | Path) -> None:
    """Plot loss, Dice, and IoU from a training history CSV."""

    history_path = Path(history_csv)
    if not history_path.is_file():
        return
    frame = pd.read_csv(history_path)
    if frame.empty:
        return

    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(frame["epoch"], frame["train_loss"], label="Train")
    axes[0].plot(frame["epoch"], frame["val_loss"], label="Validation")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(frame["epoch"], frame["train_macro_dice"], label="Train")
    axes[1].plot(frame["epoch"], frame["val_macro_dice"], label="Validation")
    axes[1].set_title("Macro Dice")
    axes[1].legend()

    axes[2].plot(frame["epoch"], frame["train_macro_iou"], label="Train")
    axes[2].plot(frame["epoch"], frame["val_macro_iou"], label="Validation")
    axes[2].set_title("Macro IoU")
    axes[2].legend()

    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_confusion_matrix(matrix: np.ndarray, output_path: str | Path) -> None:
    """Save a normalized confusion matrix."""

    normalized = matrix.astype(np.float64)
    row_sums = normalized.sum(axis=1, keepdims=True)
    normalized = np.divide(
        normalized,
        row_sums,
        out=np.zeros_like(normalized),
        where=row_sums != 0,
    )

    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(normalized, vmin=0.0, vmax=1.0)
    figure.colorbar(image, ax=axis, label="Row-normalized proportion")
    axis.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
    axis.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Confusion Matrix")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_prediction_grid(
    samples: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_path: str | Path,
) -> None:
    """Plot original OCT, ground truth, and model prediction."""

    sample_list = list(samples)
    rows = len(sample_list)
    figure, axes = plt.subplots(rows, 3, figsize=(15, max(4, 4 * rows)))
    axes = np.asarray(axes).reshape(rows, 3)

    for row, (image, target, prediction) in enumerate(sample_list):
        image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        axes[row, 0].imshow(image_uint8, cmap="gray")
        axes[row, 0].set_title("Original OCT")
        axes[row, 1].imshow(apply_overlay(image_uint8, target))
        axes[row, 1].set_title("Ground truth")
        axes[row, 2].imshow(apply_overlay(image_uint8, prediction))
        axes[row, 2].set_title("Prediction")
        for axis in axes[row]:
            axis.axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_model_comparison(
    comparison_csv: str | Path,
    output_path: str | Path,
) -> None:
    """Plot test macro Dice and IoU for all evaluated models."""

    frame = pd.read_csv(comparison_csv)
    if frame.empty:
        return
    metrics = [column for column in ("macro_dice", "macro_iou") if column in frame]
    if not metrics:
        return

    axis = frame.set_index("model_name")[metrics].plot(kind="bar", figsize=(10, 6))
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Score")
    axis.set_title("JaksuHealth Model Comparison")
    axis.grid(axis="y", alpha=0.25)
    figure = axis.get_figure()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
