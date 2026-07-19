"""Model factory for the architectures compared in JaksuHealth."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import segmentation_models_pytorch as smp
import torch
from torch import nn


_ARCHITECTURES = {
    "unet": smp.Unet,
    "unetplusplus": smp.UnetPlusPlus,
    "unet++": smp.UnetPlusPlus,
    "deeplabv3plus": smp.DeepLabV3Plus,
    "deeplabv3+": smp.DeepLabV3Plus,
}


def build_model(
    architecture: str,
    encoder_name: str = "efficientnet-b3",
    encoder_weights: str | bool | None = "imagenet",
    in_channels: int = 1,
    num_classes: int = 6,
    **model_kwargs: Any,
) -> nn.Module:
    """Build a segmentation model from a normalized architecture name."""

    key = architecture.strip().lower().replace("_", "").replace("-", "")
    aliases = {
        "unet": "unet",
        "unetplusplus": "unetplusplus",
        "unet++": "unetplusplus",
        "deeplabv3plus": "deeplabv3plus",
        "deeplabv3+": "deeplabv3plus",
    }
    normalized = aliases.get(key, key)
    if normalized not in _ARCHITECTURES:
        supported = ", ".join(sorted({"unet", "unetplusplus", "deeplabv3plus"}))
        raise ValueError(
            f"Unsupported architecture '{architecture}'. Supported: {supported}"
        )

    model_class = _ARCHITECTURES[normalized]
    return model_class(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,
        **model_kwargs,
    )


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load either a state dict or a structured JaksuHealth checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        return checkpoint
    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint, strict=strict)
        return {"model_state_dict": checkpoint}
    raise TypeError(f"Unsupported checkpoint object: {type(checkpoint)!r}")
