from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))


import argparse
import shutil

import torch
import yaml
from torch.utils.data import DataLoader

from jaksuhealth_ai.dataset import (
    OCTSegmentationDataset,
    build_train_transform,
    build_validation_transform,
)
from jaksuhealth_ai.losses import build_loss
from jaksuhealth_ai.models import build_model, count_trainable_parameters
from jaksuhealth_ai.trainer import Trainer, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a JaksuHealth model.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default=None, help="cuda, mps, cpu, or auto")
    return parser.parse_args()


def select_device(requested: str | None) -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    set_seed(int(config["seed"]))
    device = select_device(args.device)
    print(f"Using device: {device}")

    data_config = config["data"]
    output_size = (
        int(data_config["image_height"]),
        int(data_config["image_width"]),
    )
    common_dataset_kwargs = {
        "split_dir": data_config["split_dir"],
        "use_retina_crop": bool(data_config["use_retina_crop"]),
        "crop_margin": int(data_config["crop_margin"]),
        "output_size": output_size,
        "preload": bool(data_config["preload"]),
    }

    train_dataset = OCTSegmentationDataset(
        split_name=data_config["train_split"],
        transform=build_train_transform(),
        **common_dataset_kwargs,
    )
    val_dataset = OCTSegmentationDataset(
        split_name=data_config["val_split"],
        transform=build_validation_transform(),
        **common_dataset_kwargs,
    )

    training_config = config["training"]
    loader_kwargs = {
        "batch_size": int(training_config["batch_size"]),
        "num_workers": int(data_config["num_workers"]),
        "pin_memory": bool(data_config["pin_memory"] and device.type == "cuda"),
        "persistent_workers": int(data_config["num_workers"]) > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=len(train_dataset) >= int(training_config["batch_size"]),
        **loader_kwargs,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model_config = config["model"]
    model = build_model(**model_config).to(device)
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")

    loss_config = dict(config["loss"])
    loss_config["num_classes"] = int(model_config["num_classes"])
    criterion = build_loss(loss_config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(training_config["epochs"]),
        eta_min=float(training_config["min_learning_rate"]),
    )

    output_dir = Path(training_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output_dir / "config_used.yaml")

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir,
        num_classes=int(model_config["num_classes"]),
        epochs=int(training_config["epochs"]),
        accumulation_steps=int(training_config["accumulation_steps"]),
        patience=int(training_config["patience"]),
        monitor=str(training_config["monitor"]),
        gradient_clip_norm=float(training_config["gradient_clip_norm"]),
        mixed_precision=bool(training_config["mixed_precision"]),
        config=config,
    )
    summary = trainer.fit(train_loader, val_loader)
    print(summary)


if __name__ == "__main__":
    main()
