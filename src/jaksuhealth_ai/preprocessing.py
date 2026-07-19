"""Preprocessing helpers for OCT B-scans and segmentation masks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .constants import NUM_CLASSES


@dataclass(frozen=True)
class CropMetadata:
    """Coordinates needed to map a cropped prediction to the original image."""

    original_height: int
    original_width: int
    y_min: int
    y_max: int
    output_height: int
    output_width: int
    used_fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_grayscale(path: str | Path) -> np.ndarray:
    """Read an image as uint8 grayscale and raise a clear error on failure."""

    image_path = Path(path)
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read grayscale image: {image_path}")
    return image


def validate_mask_labels(
    mask: np.ndarray,
    num_classes: int = NUM_CLASSES,
    ignore_index: int = 255,
) -> None:
    """Validate that a mask contains only known class IDs or ignore_index."""

    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, received shape {mask.shape}")

    values = np.unique(mask)
    invalid = values[(values >= num_classes) & (values != ignore_index)]
    if invalid.size:
        raise ValueError(
            f"Mask contains invalid class IDs {invalid.tolist()}; "
            f"expected 0..{num_classes - 1} or {ignore_index}."
        )


def crop_retina_roi(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    output_size: tuple[int, int] = (384, 576),
    margin: int = 30,
) -> tuple[np.ndarray, np.ndarray | None, CropMetadata]:
    """Crop the dominant bright retinal band and resize it.

    Args:
        image: Grayscale OCT image with shape ``(height, width)``.
        mask: Optional integer segmentation mask aligned with ``image``.
        output_size: Output ``(height, width)``.
        margin: Number of pixels added above and below the detected band.

    Returns:
        Cropped image, cropped mask, and crop metadata. If no contour can be
        detected, the full image is resized and ``used_fallback`` is true.
    """

    if image.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, received {image.shape}")
    if mask is not None and mask.shape != image.shape:
        raise ValueError(
            f"Image and mask shapes differ: {image.shape} vs {mask.shape}"
        )

    output_height, output_width = output_size
    height, width = image.shape

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, threshold = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(
        threshold, cv2.MORPH_CLOSE, kernel, iterations=2
    )
    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        largest = max(contours, key=cv2.contourArea)
        _, y, _, band_height = cv2.boundingRect(largest)
        y_min = max(0, y - margin)
        y_max = min(height, y + band_height + margin)
        if y_max <= y_min:
            y_min, y_max = 0, height
            used_fallback = True
        else:
            used_fallback = False
    else:
        y_min, y_max = 0, height
        used_fallback = True

    image_crop = image[y_min:y_max, :]
    image_crop = cv2.resize(
        image_crop, (output_width, output_height), interpolation=cv2.INTER_LINEAR
    )

    mask_crop = None
    if mask is not None:
        mask_crop = mask[y_min:y_max, :]
        mask_crop = cv2.resize(
            mask_crop,
            (output_width, output_height),
            interpolation=cv2.INTER_NEAREST,
        )

    metadata = CropMetadata(
        original_height=height,
        original_width=width,
        y_min=y_min,
        y_max=y_max,
        output_height=output_height,
        output_width=output_width,
        used_fallback=used_fallback,
    )
    return image_crop, mask_crop, metadata


def resize_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray | None,
    output_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray | None, CropMetadata]:
    """Resize a full OCT image and optional mask without ROI detection."""

    if image.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, received {image.shape}")
    if mask is not None and mask.shape != image.shape:
        raise ValueError("Image and mask must have identical spatial dimensions")

    output_height, output_width = output_size
    height, width = image.shape
    image_out = cv2.resize(
        image, (output_width, output_height), interpolation=cv2.INTER_LINEAR
    )
    mask_out = None
    if mask is not None:
        mask_out = cv2.resize(
            mask, (output_width, output_height), interpolation=cv2.INTER_NEAREST
        )

    metadata = CropMetadata(
        original_height=height,
        original_width=width,
        y_min=0,
        y_max=height,
        output_height=output_height,
        output_width=output_width,
        used_fallback=True,
    )
    return image_out, mask_out, metadata


def inverse_crop_mask(mask: np.ndarray, metadata: CropMetadata) -> np.ndarray:
    """Map a resized ROI mask back to the original OCT image dimensions."""

    crop_height = max(1, metadata.y_max - metadata.y_min)
    restored_crop = cv2.resize(
        mask.astype(np.uint8),
        (metadata.original_width, crop_height),
        interpolation=cv2.INTER_NEAREST,
    )
    restored = np.zeros(
        (metadata.original_height, metadata.original_width), dtype=np.uint8
    )
    restored[metadata.y_min : metadata.y_max, :] = restored_crop[
        : metadata.y_max - metadata.y_min, :
    ]
    return restored


def normalize_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert uint8 grayscale pixels to float32 in the [0, 1] range."""

    return image.astype(np.float32) / 255.0
