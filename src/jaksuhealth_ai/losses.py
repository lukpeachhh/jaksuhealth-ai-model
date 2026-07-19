"""Loss functions for class-imbalanced multi-class segmentation."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class MulticlassFocalLoss(nn.Module):
    """Multi-class focal loss with optional class weighting."""

    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: Sequence[float] | torch.Tensor | None = None,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()

        if gamma < 0:
            raise ValueError("gamma must be greater than or equal to zero.")

        self.gamma = gamma
        self.ignore_index = ignore_index

        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            weights = torch.as_tensor(
                class_weights,
                dtype=torch.float32,
            )

            if weights.ndim != 1:
                raise ValueError("class_weights must be one-dimensional.")

            self.register_buffer("class_weights", weights)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate focal loss from unweighted class probabilities."""

        if logits.ndim != 4:
            raise ValueError(
                "logits must have shape [batch, classes, height, width]."
            )

        if targets.ndim != 3:
            raise ValueError(
                "targets must have shape [batch, height, width]."
            )

        if logits.shape[0] != targets.shape[0]:
            raise ValueError(
                "logits and targets must have the same batch size."
            )

        if logits.shape[2:] != targets.shape[1:]:
            raise ValueError(
                "logits and targets must have the same spatial dimensions."
            )

        num_classes = logits.shape[1]

        if (
            self.class_weights is not None
            and len(self.class_weights) != num_classes
        ):
            raise ValueError(
                "The number of class weights must match "
                f"the number of classes: {num_classes}."
            )

        valid_mask = targets != self.ignore_index

        if not valid_mask.any():
            return logits.sum() * 0.0

        safe_targets = targets.masked_fill(~valid_mask, 0).long()

        log_probabilities = F.log_softmax(
            logits,
            dim=1,
        )

        log_pt = log_probabilities.gather(
            dim=1,
            index=safe_targets.unsqueeze(1),
        ).squeeze(1)

        pt = log_pt.exp()

        focal_factor = (1.0 - pt).pow(self.gamma)
        focal_loss = -focal_factor * log_pt

        if self.class_weights is not None:
            pixel_weights = self.class_weights[safe_targets]
            focal_loss = focal_loss * pixel_weights

        return focal_loss[valid_mask].mean()


class MulticlassTverskyLoss(nn.Module):
    """Calculate Tversky loss independently for each class."""

    def __init__(
        self,
        num_classes: int,
        alpha: float = 0.5,
        beta: float = 0.5,
        ignore_index: int = 255,
        include_background: bool = False,
        smooth: float = 1e-7,
    ) -> None:
        super().__init__()

        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")

        if alpha < 0 or beta < 0:
            raise ValueError(
                "alpha and beta must be greater than or equal to zero."
            )

        if alpha + beta <= 0:
            raise ValueError(
                "The sum of alpha and beta must be greater than zero."
            )

        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.ignore_index = ignore_index
        self.include_background = include_background
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate mean Tversky loss across selected classes."""

        valid_mask = targets != self.ignore_index

        if not valid_mask.any():
            return logits.sum() * 0.0

        safe_targets = targets.masked_fill(
            ~valid_mask,
            0,
        ).long()

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        one_hot_targets = F.one_hot(
            safe_targets,
            num_classes=self.num_classes,
        ).permute(0, 3, 1, 2).to(dtype=probabilities.dtype)

        valid_float = valid_mask.unsqueeze(1).to(
            dtype=probabilities.dtype
        )

        probabilities = probabilities * valid_float
        one_hot_targets = one_hot_targets * valid_float

        reduction_dimensions = (0, 2, 3)

        true_positive = (
            probabilities * one_hot_targets
        ).sum(reduction_dimensions)

        false_positive = (
            probabilities * (1.0 - one_hot_targets)
        ).sum(reduction_dimensions)

        false_negative = (
            (1.0 - probabilities) * one_hot_targets
        ).sum(reduction_dimensions)

        tversky_score = (
            true_positive + self.smooth
        ) / (
            true_positive
            + self.alpha * false_positive
            + self.beta * false_negative
            + self.smooth
        )

        if not self.include_background:
            tversky_score = tversky_score[1:]

        return 1.0 - tversky_score.mean()


class FocalTverskyLoss(nn.Module):
    """Combine multi-class focal loss and Tversky loss."""

    def __init__(
        self,
        num_classes: int,
        class_weights: Sequence[float] | torch.Tensor | None = None,
        focal_gamma: float = 2.0,
        tversky_alpha: float = 0.5,
        tversky_beta: float = 0.5,
        focal_weight: float = 1.0,
        tversky_weight: float = 1.0,
        ignore_index: int = 255,
        include_background_in_tversky: bool = False,
    ) -> None:
        super().__init__()

        if focal_weight < 0 or tversky_weight < 0:
            raise ValueError(
                "Loss component weights cannot be negative."
            )

        if focal_weight + tversky_weight <= 0:
            raise ValueError(
                "At least one loss component must have a positive weight."
            )

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

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the weighted combination of both losses."""

        focal_loss = self.focal(
            logits,
            targets,
        )

        tversky_loss = self.tversky(
            logits,
            targets,
        )

        return (
            self.focal_weight * focal_loss
            + self.tversky_weight * tversky_loss
        )


def build_loss(config: dict) -> nn.Module:
    """Build a segmentation loss function from configuration."""

    loss_name = str(
        config.get("name", "focal_tversky")
    ).lower()

    if loss_name != "focal_tversky":
        raise ValueError(
            "Unsupported loss function. "
            "Only 'focal_tversky' is currently supported."
        )

    return FocalTverskyLoss(
        num_classes=int(config.get("num_classes", 6)),
        class_weights=config.get("class_weights"),
        focal_gamma=float(config.get("focal_gamma", 2.0)),
        tversky_alpha=float(
            config.get("tversky_alpha", 0.5)
        ),
        tversky_beta=float(
            config.get("tversky_beta", 0.5)
        ),
        focal_weight=float(config.get("focal_weight", 1.0)),
        tversky_weight=float(
            config.get("tversky_weight", 1.0)
        ),
        ignore_index=int(config.get("ignore_index", 255)),
        include_background_in_tversky=bool(
            config.get(
                "include_background_in_tversky",
                False,
            )
        ),
    )