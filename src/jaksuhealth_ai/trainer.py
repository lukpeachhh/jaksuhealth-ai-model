"""Reusable training loop for JaksuHealth segmentation experiments."""

from __future__ import annotations

import csv
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import MetricReport, SegmentationConfusionMatrix
from .visualization import plot_training_curves


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class Trainer:
    """Train a segmentation model with AMP, accumulation, and early stopping."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        device: torch.device,
        output_dir: str | Path,
        num_classes: int,
        epochs: int = 60,
        accumulation_steps: int = 1,
        patience: int = 15,
        monitor: str = "macro_iou",
        gradient_clip_norm: float | None = 1.0,
        mixed_precision: bool = True,
        config: dict | None = None,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_classes = num_classes
        self.epochs = epochs
        self.accumulation_steps = max(1, accumulation_steps)
        self.patience = patience
        self.monitor = monitor
        self.gradient_clip_norm = gradient_clip_norm
        self.amp_enabled = mixed_precision and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.config = config or {}
        self.history: list[dict[str, float | int]] = []

    def _run_epoch(
        self,
        loader: DataLoader,
        training: bool,
        epoch: int,
    ) -> tuple[float, MetricReport]:
        self.model.train(training)
        confusion = SegmentationConfusionMatrix(num_classes=self.num_classes)
        running_loss = 0.0
        sample_count = 0

        if training:
            self.optimizer.zero_grad(set_to_none=True)

        label = "train" if training else "val"
        progress = tqdm(loader, desc=f"Epoch {epoch}/{self.epochs} [{label}]")

        context = torch.enable_grad() if training else torch.inference_mode()
        with context:
            for step, (images, masks) in enumerate(progress, start=1):
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                with torch.amp.autocast(
                    device_type=self.device.type,
                    enabled=self.amp_enabled,
                ):
                    logits = self.model(images)
                    raw_loss = self.criterion(logits, masks)
                    loss = raw_loss / self.accumulation_steps

                if training:
                    self.scaler.scale(loss).backward()
                    should_step = (
                        step % self.accumulation_steps == 0 or step == len(loader)
                    )
                    if should_step:
                        self.scaler.unscale_(self.optimizer)
                        if self.gradient_clip_norm is not None:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.gradient_clip_norm
                            )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)

                batch_size = images.shape[0]
                running_loss += float(raw_loss.detach().cpu()) * batch_size
                sample_count += batch_size
                predictions = torch.argmax(logits.detach(), dim=1)
                confusion.update(predictions, masks)
                progress.set_postfix(loss=f"{float(raw_loss.detach()):.4f}")

        return running_loss / max(sample_count, 1), confusion.compute()

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> dict[str, Any]:
        """Train and save the best checkpoint according to the monitor metric."""

        best_value = float("-inf")
        best_epoch = 0
        no_improvement = 0
        best_state: dict[str, torch.Tensor] | None = None

        for epoch in range(1, self.epochs + 1):
            train_loss, train_report = self._run_epoch(train_loader, True, epoch)
            val_loss, val_report = self._run_epoch(val_loader, False, epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            row: dict[str, float | int] = {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            row.update({f"train_{k}": v for k, v in train_report.overall.items()})
            row.update({f"val_{k}": v for k, v in val_report.overall.items()})
            self.history.append(row)
            self._save_history()

            monitor_key = f"val_{self.monitor}"
            if monitor_key not in row:
                raise KeyError(
                    f"Monitor '{self.monitor}' is unavailable. "
                    f"Available validation keys: {[k for k in row if k.startswith('val_')]}"
                )
            current_value = float(row[monitor_key])

            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, {monitor_key}={current_value:.4f}"
            )

            if current_value > best_value:
                best_value = current_value
                best_epoch = epoch
                no_improvement = 0
                best_state = deepcopy(self.model.state_dict())
                self._save_checkpoint(best_state, epoch, best_value)
                self._save_validation_report(val_report)
            else:
                no_improvement += 1
                if no_improvement >= self.patience:
                    print(
                        f"Early stopping at epoch {epoch}; "
                        f"best epoch was {best_epoch}."
                    )
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        plot_training_curves(
            self.output_dir / "history.csv",
            self.output_dir / "training_curves.png",
        )
        summary = {
            "best_epoch": best_epoch,
            "best_monitor": self.monitor,
            "best_value": best_value,
        }
        (self.output_dir / "training_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary

    def _save_history(self) -> None:
        path = self.output_dir / "history.csv"
        if not self.history:
            return
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.history[0].keys()))
            writer.writeheader()
            writer.writerows(self.history)

    def _save_checkpoint(
        self,
        state_dict: dict[str, torch.Tensor],
        epoch: int,
        monitor_value: float,
    ) -> None:
        torch.save(
            {
                "model_state_dict": state_dict,
                "epoch": epoch,
                "monitor": self.monitor,
                "monitor_value": monitor_value,
                "config": self.config,
            },
            self.output_dir / "best_model.pth",
        )

    def _save_validation_report(self, report: MetricReport) -> None:
        report.per_class.to_csv(
            self.output_dir / "best_val_per_class.csv", index=False
        )
        np.save(
            self.output_dir / "best_val_confusion_matrix.npy",
            report.confusion_matrix,
        )
        (self.output_dir / "best_val_overall.json").write_text(
            json.dumps(report.overall, indent=2), encoding="utf-8"
        )
