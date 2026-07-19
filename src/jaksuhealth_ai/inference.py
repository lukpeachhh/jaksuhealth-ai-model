"""Single-image inference used by scripts and software integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn

from .constants import CLASS_NAMES_BY_ID
from .evaluator import predict_logits
from .preprocessing import (
    CropMetadata,
    crop_retina_roi,
    inverse_crop_mask,
    normalize_grayscale,
    read_grayscale,
    resize_image_and_mask,
)


def predict_array(
    model: nn.Module,
    image: np.ndarray,
    device: torch.device,
    output_size: tuple[int, int] = (384, 576),
    use_retina_crop: bool = True,
    crop_margin: int = 30,
    use_tta: bool = True,
) -> dict[str, Any]:
    """Run segmentation on a grayscale NumPy OCT image."""

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim != 2:
        raise ValueError(f"Expected a grayscale or BGR image, got {image.shape}")

    if use_retina_crop:
        processed, _, metadata = crop_retina_roi(
            image, None, output_size=output_size, margin=crop_margin
        )
    else:
        processed, _, metadata = resize_image_and_mask(image, None, output_size)

    tensor = torch.from_numpy(normalize_grayscale(processed)[None, None, ...])
    tensor = tensor.float().to(device)

    model.eval()
    with torch.inference_mode():
        logits = predict_logits(model, tensor, use_tta=use_tta)
        probabilities = torch.softmax(logits, dim=1)
        processed_mask = torch.argmax(probabilities, dim=1)[0].cpu().numpy()

    restored_mask = inverse_crop_mask(processed_mask, metadata)
    total_pixels = restored_mask.size
    lesion_statistics: dict[str, dict[str, float | int]] = {}
    for class_id, class_name in CLASS_NAMES_BY_ID.items():
        if class_id == 0:
            continue
        pixels = int(np.count_nonzero(restored_mask == class_id))
        lesion_statistics[class_name] = {
            "pixels": pixels,
            "percentage": 100.0 * pixels / max(total_pixels, 1),
        }

    return {
        "mask": restored_mask,
        "processed_mask": processed_mask,
        "processed_image": processed,
        "metadata": metadata,
        "lesion_statistics": lesion_statistics,
    }


def predict_file(
    model: nn.Module,
    image_path: str | Path,
    device: torch.device,
    **kwargs: Any,
) -> dict[str, Any]:
    """Read an OCT image and call :func:`predict_array`."""

    image = read_grayscale(image_path)
    result = predict_array(model, image, device, **kwargs)
    result["original_image"] = image
    return result
