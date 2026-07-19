from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))


import argparse
import json

import pandas as pd

from jaksuhealth_ai.visualization import plot_model_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare evaluated models.")
    parser.add_argument("--runs-dir", type=Path, default=Path("results/runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--evaluation-folder", default="evaluation_test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_rows = []
    class_frames = []

    for run_dir in sorted(path for path in args.runs_dir.iterdir() if path.is_dir()):
        evaluation_dir = run_dir / args.evaluation_folder
        overall_path = evaluation_dir / "overall_metrics.json"
        per_class_path = evaluation_dir / "per_class_metrics.csv"
        if not overall_path.is_file() or not per_class_path.is_file():
            continue

        summary = json.loads(overall_path.read_text(encoding="utf-8"))
        summary_rows.append(summary)
        per_class = pd.read_csv(per_class_path)
        per_class.insert(0, "model_name", summary["model_name"])
        class_frames.append(per_class)

    if not summary_rows:
        raise FileNotFoundError(
            f"No evaluation outputs found under {args.runs_dir}. "
            "Run scripts/evaluate.py for each model first."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    comparison = pd.DataFrame(summary_rows).sort_values(
        "macro_iou", ascending=False
    )
    comparison_path = args.output_dir / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    pd.concat(class_frames, ignore_index=True).to_csv(
        args.output_dir / "per_class_metrics.csv", index=False
    )
    plot_model_comparison(
        comparison_path, figures_dir / "model_comparison.png"
    )
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
