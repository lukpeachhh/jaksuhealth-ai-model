from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIGS = (
    Path("configs/unet.yaml"),
    Path("configs/unetplusplus.yaml"),
    Path("configs/deeplabv3plus.yaml"),
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Train, evaluate, and compare all configured JaksuHealth "
            "segmentation models."
        )
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        type=Path,
        default=list(DEFAULT_CONFIGS),
        help="Configuration files to process in order.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Execution device passed to training and evaluation.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split evaluated after training. Default: test.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and use existing best_model.pth checkpoints.",
    )
    parser.add_argument(
        "--skip-evaluate",
        action="store_true",
        help="Skip model evaluation.",
    )
    parser.add_argument(
        "--skip-compare",
        action="store_true",
        help="Skip generation of combined comparison files.",
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
            "masks during evaluation."
        ),
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("results/runs"),
        help="Directory searched by the model-comparison script.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for combined model-comparison outputs.",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    """Run a subprocess and stop immediately if it fails."""

    print("\n$", " ".join(command), flush=True)
    subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=True,
    )


def read_config(config_path: Path) -> dict[str, Any]:
    """Load and validate an experiment configuration."""

    resolved_path = (
        config_path
        if config_path.is_absolute()
        else REPOSITORY_ROOT / config_path
    )

    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {resolved_path}"
        )

    config = yaml.safe_load(
        resolved_path.read_text(encoding="utf-8")
    )

    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration must contain a YAML mapping: {resolved_path}"
        )

    required_sections = {
        "project",
        "training",
    }
    missing_sections = required_sections.difference(config)

    if missing_sections:
        missing = ", ".join(sorted(missing_sections))
        raise KeyError(
            f"Missing configuration sections in {resolved_path}: {missing}"
        )

    return config


def repository_relative(path: Path) -> Path:
    """Return a path relative to the repository when possible."""

    resolved_path = path.resolve()

    try:
        return resolved_path.relative_to(REPOSITORY_ROOT)
    except ValueError:
        return resolved_path


def resolve_output_dir(config: dict[str, Any]) -> Path:
    """Resolve the training output directory from a configuration."""

    output_dir = Path(config["training"]["output_dir"])

    if not output_dir.is_absolute():
        output_dir = REPOSITORY_ROOT / output_dir

    return output_dir


def experiment_name(config: dict[str, Any], config_path: Path) -> str:
    """Read the experiment name, falling back to the config filename."""

    project_config = config.get("project", {})
    name = str(project_config.get("experiment_name", "")).strip()

    return name or config_path.stem


def train_model(
    config_path: Path,
    device: str,
) -> None:
    """Run model training for one configuration."""

    run_command(
        [
            sys.executable,
            "scripts/train.py",
            "--config",
            str(repository_relative(config_path)),
            "--device",
            device,
        ]
    )


def evaluate_model(
    config_path: Path,
    checkpoint_path: Path,
    split: str,
    device: str,
    no_tta: bool,
    save_predictions: bool,
) -> None:
    """Run test evaluation for one trained model."""

    command = [
        sys.executable,
        "scripts/evaluate.py",
        "--config",
        str(repository_relative(config_path)),
        "--checkpoint",
        str(repository_relative(checkpoint_path)),
        "--split",
        split,
        "--device",
        device,
    ]

    if no_tta:
        command.append("--no-tta")

    if save_predictions:
        command.append("--save-predictions")

    run_command(command)


def compare_models(
    runs_dir: Path,
    output_dir: Path,
    split: str,
) -> None:
    """Generate combined model-comparison CSV files and figures."""

    run_command(
        [
            sys.executable,
            "scripts/compare_models.py",
            "--runs-dir",
            str(runs_dir),
            "--output-dir",
            str(output_dir),
            "--evaluation-folder",
            f"evaluation_{split.lower()}",
        ]
    )


def main() -> None:
    """Execute the requested experiment workflow."""

    args = parse_args()

    if args.skip_evaluate and args.save_predictions:
        raise ValueError(
            "--save-predictions cannot be used together with "
            "--skip-evaluate."
        )

    experiment_summaries: list[tuple[str, Path, Path]] = []

    for supplied_config_path in args.configs:
        config_path = (
            supplied_config_path
            if supplied_config_path.is_absolute()
            else REPOSITORY_ROOT / supplied_config_path
        )
        config_path = config_path.resolve()
        config = read_config(config_path)

        name = experiment_name(
            config=config,
            config_path=config_path,
        )
        output_dir = resolve_output_dir(config)
        checkpoint_path = output_dir / "best_model.pth"

        experiment_summaries.append(
            (
                name,
                output_dir,
                checkpoint_path,
            )
        )

        print("\n" + "=" * 72)
        print(f"Experiment: {name}")
        print(f"Config: {config_path}")
        print(f"Output directory: {output_dir}")
        print(f"Checkpoint: {checkpoint_path}")
        print("=" * 72)

        if not args.skip_train:
            train_model(
                config_path=config_path,
                device=args.device,
            )

        if args.skip_evaluate:
            continue

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found for experiment '{name}': "
                f"{checkpoint_path}. Train the model first or remove "
                "--skip-train."
            )

        evaluate_model(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            split=args.split,
            device=args.device,
            no_tta=args.no_tta,
            save_predictions=args.save_predictions,
        )

    if not args.skip_compare:
        compare_models(
            runs_dir=args.runs_dir,
            output_dir=args.output_dir,
            split=args.split,
        )

    print("\nExperiment workflow completed.")

    for name, output_dir, checkpoint_path in experiment_summaries:
        print(f"- {name}")
        print(f"  output: {output_dir}")
        print(f"  checkpoint: {checkpoint_path}")

    if args.save_predictions:
        print(
            "\nSaved evaluation images, ground-truth masks, and "
            "predictions for each evaluated model."
        )


if __name__ == "__main__":
    main()