"""Loss functions used for class-imbalanced multi-class segmentation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MulticlassFocalLoss(nn.Module):
    """Weighted focal cross-entropy with ignore-index support."""

    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: list[float] | torch.Tensor | None = None,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            weights = torch.as_tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cross_entropy = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        valid = targets != self.ignore_index
        if not torch.any(valid):
            return logits.sum() * 0.0
        ce_valid = cross_entropy[valid]
        probability = torch.exp(-ce_valid)
        return (((1.0 - probability) ** self.gamma) * ce_valid).mean()


class MulticlassTverskyLoss(nn.Module):
    """Differentiable Tversky loss computed per foreground class."""

    def __init__(
        self,
        num_classes: int,
        alpha: float = 0.3,
        beta: float = 0.7,
        ignore_index: int = 255,
        include_background: bool = False,
        smooth: float = 1e-7,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.ignore_index = ignore_index
        self.include_background = include_background
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(logits, dim=1)
        valid = targets != self.ignore_index
        safe_targets = targets.masked_fill(~valid, 0)
        one_hot = F.one_hot(
            safe_targets, num_classes=self.num_classes
        ).permute(0, 3, 1, 2).float()

        valid_float = valid.unsqueeze(1).float()
        probabilities = probabilities * valid_float
        one_hot = one_hot * valid_float

        dims = (0, 2, 3)
        true_positive = (probabilities * one_hot).sum(dims)
        false_positive = (probabilities * (1.0 - one_hot)).sum(dims)
        false_negative = ((1.0 - probabilities) * one_hot).sum(dims)

        score = (true_positive + self.smooth) / (
            true_positive
            + self.alpha * false_positive
            + self.beta * false_negative
            + self.smooth
        )
        if not self.include_background:
            score = score[1:]
        return 1.0 - score.mean()


class FocalTverskyLoss(nn.Module):
    """Linear combination of weighted focal loss and Tversky loss."""

    def __init__(
        self,
        num_classes: int,
        class_weights: list[float] | None = None,
        focal_gamma: float = 2.0,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        focal_weight: float = 1.0,
        tversky_weight: float = 1.0,
        ignore_index: int = 255,
        include_background_in_tversky: bool = False,
    ) -> None:
        super().__init__()
        self.focal = MulticlassFocalLoss(
            gamma=focal_gamma,
            class_weights=class_weights,
            ignore_index=ignore_index,
        )
        self.tversky = MulticlassTverskyLoss(
            num_classes=num_classes,
            alpha=tversky_alpha,
            beta=tversky_beta,
            ignore_index=ignore_index,
            include_background=include_background_in_tversky,
        )
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.focal_weight * self.focal(logits, targets)
            + self.tversky_weight * self.tversky(logits, targets)
        )


def build_loss(config: dict) -> nn.Module:
    """Build the configured loss function."""

    name = str(config.get("name", "focal_tversky")).lower()
    if name != "focal_tversky":
        raise ValueError("Only 'focal_tversky' is currently supported")
    return FocalTverskyLoss(
        num_classes=int(config.get("num_classes", 6)),
        class_weights=config.get("class_weights"),
        focal_gamma=float(config.get("focal_gamma", 2.0)),
        tversky_alpha=float(config.get("tversky_alpha", 0.3)),
        tversky_beta=float(config.get("tversky_beta", 0.7)),
        focal_weight=float(config.get("focal_weight", 1.0)),
        tversky_weight=float(config.get("tversky_weight", 1.0)),
        ignore_index=int(config.get("ignore_index", 255)),
        include_background_in_tversky=bool(
            config.get("include_background_in_tversky", False)
        ),
    )
