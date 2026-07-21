from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY = REPOSITORY_ROOT / "src"

if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from jaksuhealth_ai.dataset import (  # noqa: E402
    OCTSegmentationDataset,
    build_validation_transform,
)
from jaksuhealth_ai.evaluator import evaluate_model  # noqa: E402
from jaksuhealth_ai.models import build_model, load_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Evaluate a trained OCT segmentation model."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML experiment configuration.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help=(
            "Dataset split to evaluate. Standard aliases are "
            "'train', 'val', and 'test'."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device, for example 'cuda', 'mps', or 'cpu'.",
    )
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Disable horizontal-flip test-time augmentation.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help=(
            "Save processed OCT images, ground-truth masks, and predicted "
            "masks as PNG files."
        ),
    )
    return parser.parse_args()


def read_config(config_path: Path) -> dict[str, Any]:
    """Read and validate a YAML configuration file."""

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration must contain a YAML mapping: {config_path}"
        )

    required_sections = {
        "project",
        "data",
        "model",
        "training",
        "evaluation",
    }
    missing_sections = required_sections.difference(config)

    if missing_sections:
        missing = ", ".join(sorted(missing_sections))
        raise KeyError(f"Missing configuration sections: {missing}")

    return config


def select_device(requested: str | None) -> torch.device:
    """Select the requested accelerator or the best available device."""

    if requested is not None:
        device = torch.device(requested)

        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but no CUDA-compatible GPU is available."
            )

        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS was requested, but Apple Metal acceleration is unavailable."
            )

        return device

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def resolve_split_name(
    requested_split: str,
    data_config: dict[str, Any],
) -> str:
    """Resolve train/validation/test aliases from the configuration."""

    aliases = {
        "train": str(data_config["train_split"]),
        "val": str(data_config["val_split"]),
        "validation": str(data_config["val_split"]),
        "test": str(data_config["test_split"]),
    }

    return aliases.get(requested_split.lower(), requested_split)


def build_evaluation_loader(
    config: dict[str, Any],
    split_name: str,
    device: torch.device,
    return_metadata: bool,
) -> DataLoader:
    """Build a deterministic DataLoader for evaluation."""

    data_config = config["data"]
    training_config = config["training"]

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
        return_metadata=return_metadata,
    )

    num_workers = int(data_config["num_workers"])

    return DataLoader(
        dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(
            data_config["pin_memory"] and device.type == "cuda"
        ),
        persistent_workers=num_workers > 0,
    )


def main() -> None:
    """Load a checkpoint, evaluate it, and export evaluation artifacts."""

    args = parse_args()
    config = read_config(args.config)

    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint file not found: {args.checkpoint}"
        )

    device = select_device(args.device)
    model_config = config["model"]
    data_config = config["data"]
    evaluation_config = config["evaluation"]
    training_config = config["training"]

    split_name = resolve_split_name(
        requested_split=args.split,
        data_config=data_config,
    )

    model = build_model(**model_config).to(device)
    checkpoint = load_checkpoint(
        model=model,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    loader = build_evaluation_loader(
        config=config,
        split_name=split_name,
        device=device,
        return_metadata=args.save_predictions,
    )

    model_name = str(config["project"]["experiment_name"])
    output_dir = (
        Path(training_config["output_dir"])
        / f"evaluation_{args.split.lower()}"
    )

    use_tta = bool(
        evaluation_config.get("use_tta", False)
        and not args.no_tta
    )

    print(f"Model: {model_name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Dataset split: {split_name}")
    print(f"Device: {device}")
    print(f"TTA enabled: {use_tta}")
    print(f"Save predictions: {args.save_predictions}")
    print(f"Output directory: {output_dir}")

    result = evaluate_model(
        model=model,
        loader=loader,
        device=device,
        output_dir=output_dir,
        model_name=model_name,
        num_classes=int(model_config["num_classes"]),
        use_tta=use_tta,
        save_samples=int(evaluation_config.get("save_samples", 8)),
        save_predictions=args.save_predictions,
    )

    print("Evaluation completed.")
    print(yaml.safe_dump(result["summary"], sort_keys=False))


if __name__ == "__main__":
    main()