"""Run training, test evaluation, and model comparison sequentially."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_CONFIGS = (
    Path("configs/unet.yaml"),
    Path("configs/unetplusplus.yaml"),
    Path("configs/deeplabv3plus.yaml"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all JaksuHealth segmentation experiments."
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        type=Path,
        default=list(DEFAULT_CONFIGS),
        help="Configuration files to run in order.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--no-tta", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("\n$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def experiment_name(config_path: Path) -> str:
    return config_path.stem


def main() -> None:
    args = parse_args()

    for config_path in args.configs:
        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration not found: {config_path}")

        name = experiment_name(config_path)
        checkpoint = Path("results/runs") / name / "best_model.pth"

        if not args.skip_train:
            run(
                [
                    sys.executable,
                    "scripts/train.py",
                    "--config",
                    str(config_path),
                    "--device",
                    args.device,
                ]
            )

        if not args.skip_evaluate:
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"Checkpoint not found for {name}: {checkpoint}"
                )
            command = [
                sys.executable,
                "scripts/evaluate.py",
                "--config",
                str(config_path),
                "--checkpoint",
                str(checkpoint),
                "--split",
                "test",
                "--device",
                args.device,
            ]
            if args.no_tta:
                command.append("--no-tta")
            run(command)

    if not args.skip_evaluate:
        run(
            [
                sys.executable,
                "scripts/compare_models.py",
                "--runs-dir",
                "results/runs",
                "--output-dir",
                "results",
            ]
        )


if __name__ == "__main__":
    main()
