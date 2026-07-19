from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))


import argparse
import json

import cv2
import torch
import yaml

from jaksuhealth_ai.inference import predict_file
from jaksuhealth_ai.models import build_model, load_checkpoint
from jaksuhealth_ai.visualization import apply_overlay, colorize_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict one OCT image.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    model = build_model(**config["model"]).to(device)
    load_checkpoint(model, args.checkpoint, device=device)

    data_config = config["data"]
    result = predict_file(
        model=model,
        image_path=args.image,
        device=device,
        output_size=(
            int(data_config["image_height"]),
            int(data_config["image_width"]),
        ),
        use_retina_crop=bool(data_config["use_retina_crop"]),
        crop_margin=int(data_config["crop_margin"]),
        use_tta=bool(config["evaluation"]["use_tta"] and not args.no_tta),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    original = result["original_image"]
    mask = result["mask"].astype("uint8")
    overlay = apply_overlay(original, mask)

    cv2.imwrite(str(args.output_dir / "predicted_mask.png"), mask)
    cv2.imwrite(
        str(args.output_dir / "predicted_mask_color.png"),
        cv2.cvtColor(colorize_mask(mask), cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(args.output_dir / "prediction_overlay.png"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
    )
    (args.output_dir / "lesion_statistics.json").write_text(
        json.dumps(result["lesion_statistics"], indent=2), encoding="utf-8"
    )
    print(json.dumps(result["lesion_statistics"], indent=2))


if __name__ == "__main__":
    main()
