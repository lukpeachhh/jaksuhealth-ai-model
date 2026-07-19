from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))


import argparse

import torch
import yaml
from torch.utils.data import DataLoader

from jaksuhealth_ai.dataset import OCTSegmentationDataset, build_validation_transform
from jaksuhealth_ai.evaluator import evaluate_model
from jaksuhealth_ai.models import build_model, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-tta", action="store_true")
    return parser.parse_args()


def select_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = select_device(args.device)
    model_config = config["model"]
    model = build_model(**model_config).to(device)
    load_checkpoint(model, args.checkpoint, device=device)

    data_config = config["data"]
    split_name = {
        "train": data_config["train_split"],
        "val": data_config["val_split"],
        "test": data_config["test_split"],
    }.get(args.split, args.split)

    dataset = OCTSegmentationDataset(
        split_dir=data_config["split_dir"],
        split_name=split_name,
        transform=build_validation_transform(),
        use_retina_crop=bool(data_config["use_retina_crop"]),
        crop_margin=int(data_config["crop_margin"]),
        output_size=(
            int(data_config["image_height"]),
            int(data_config["image_width"]),
        ),
        preload=bool(data_config["preload"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(data_config["num_workers"]),
        pin_memory=bool(data_config["pin_memory"] and device.type == "cuda"),
        persistent_workers=int(data_config["num_workers"]) > 0,
    )

    model_name = config["project"]["experiment_name"]
    output_dir = Path(config["training"]["output_dir"]) / f"evaluation_{args.split}"
    result = evaluate_model(
        model=model,
        loader=loader,
        device=device,
        output_dir=output_dir,
        model_name=model_name,
        num_classes=int(model_config["num_classes"]),
        use_tta=bool(config["evaluation"]["use_tta"] and not args.no_tta),
        save_samples=int(config["evaluation"]["save_samples"]),
    )
    print(result["summary"])


if __name__ == "__main__":
    main()
