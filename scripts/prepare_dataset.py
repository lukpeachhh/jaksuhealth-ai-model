from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))


import argparse
import csv
import json
import os
import random
import shutil
from collections import defaultdict

import cv2
import numpy as np

from jaksuhealth_ai.constants import CLASS_NAMES_BY_ID, SUPPORTED_IMAGE_EXTENSIONS
from jaksuhealth_ai.preprocessing import read_grayscale, validate_mask_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a deterministic grouped train/val/test split."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--group-delimiter",
        "--patient-delimiter",
        dest="group_delimiter",
        default="_",
        help=(
            "Delimiter used to extract the group ID from each filename. "
            "For AMD-SD names such as n_x.png, n is the eye ID."
        ),
    )
    parser.add_argument("--mode", choices=("copy", "symlink"), default="copy")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def group_id_from_name(filename: str, delimiter: str) -> str:
    stem = Path(filename).stem
    group_id = stem.split(delimiter, maxsplit=1)[0]
    if not group_id:
        raise ValueError(f"Could not extract group ID from {filename}")
    return group_id


def calculate_capacities(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {name: total * ratio for name, ratio in ratios.items()}
    capacities = {name: int(value) for name, value in raw.items()}
    remaining = total - sum(capacities.values())
    order = sorted(ratios, key=lambda name: raw[name] - capacities[name], reverse=True)
    for name in order[:remaining]:
        capacities[name] += 1
    return capacities


def balanced_group_split(
    group_stats: dict[str, np.ndarray],
    ratios: dict[str, float],
    seed: int,
) -> dict[str, list[str]]:
    group_ids = list(group_stats)
    capacities = calculate_capacities(len(group_ids), ratios)
    total_class_pixels = sum(group_stats.values(), start=np.zeros(5, dtype=np.float64))
    targets = {name: total_class_pixels * ratio for name, ratio in ratios.items()}
    current = {name: np.zeros(5, dtype=np.float64) for name in ratios}
    splits: dict[str, list[str]] = {name: [] for name in ratios}

    random_generator = random.Random(seed)
    random_generator.shuffle(group_ids)
    group_ids.sort(key=lambda gid: float(group_stats[gid].sum()), reverse=True)

    def global_error(candidate_split: str, features: np.ndarray) -> float:
        error = 0.0
        for split_name in ratios:
            values = current[split_name] + (
                features if split_name == candidate_split else 0.0
            )
            denominator = targets[split_name] + 1.0
            error += float(np.mean(((values - targets[split_name]) / denominator) ** 2))
        return error

    for group_id in group_ids:
        features = group_stats[group_id]
        candidates = [
            name for name in ratios if len(splits[name]) < capacities[name]
        ]
        chosen = min(candidates, key=lambda name: global_error(name, features))
        splits[chosen].append(group_id)
        current[chosen] += features

    return splits


def transfer(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
    else:
        os.symlink(source.resolve(), destination)


def main() -> None:
    args = parse_args()
    ratios = {
        "train": args.train_ratio,
        "val": args.val_ratio,
        "test": args.test_ratio,
    }
    if not np.isclose(sum(ratios.values()), 1.0):
        raise ValueError("Train, validation, and test ratios must sum to 1.0")

    image_dir = args.data_dir / "images"
    mask_dir = args.data_dir / "masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError("Expected data-dir/images and data-dir/masks")

    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    pairs_by_group: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    group_stats: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(5, dtype=np.float64)
    )

    for image_path in image_paths:
        mask_path = mask_dir / image_path.name
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing mask for {image_path.name}")
        image = read_grayscale(image_path)
        mask = read_grayscale(mask_path)
        if image.shape != mask.shape:
            raise ValueError(f"Shape mismatch for {image_path.name}")
        validate_mask_labels(mask)
        group_id = group_id_from_name(image_path.name, args.group_delimiter)
        pairs_by_group[group_id].append((image_path, mask_path))
        group_stats[group_id] += np.asarray(
            [np.count_nonzero(mask == class_id) for class_id in range(1, 6)],
            dtype=np.float64,
        )

    splits = balanced_group_split(group_stats, ratios, args.seed)

    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory exists: {args.output_dir}. Use --overwrite."
            )
        shutil.rmtree(args.output_dir)

    for split_name, group_ids in splits.items():
        for group_id in group_ids:
            for image_path, mask_path in pairs_by_group[group_id]:
                transfer(
                    image_path,
                    args.output_dir / split_name / "images" / image_path.name,
                    args.mode,
                )
                transfer(
                    mask_path,
                    args.output_dir / split_name / "masks" / mask_path.name,
                    args.mode,
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "split_groups.json").write_text(
        json.dumps(splits, indent=2), encoding="utf-8"
    )

    summary_rows = []
    for split_name, group_ids in splits.items():
        pixels = sum(
            (group_stats[group_id] for group_id in group_ids),
            start=np.zeros(5, dtype=np.float64),
        )
        row = {
            "split": split_name,
            "groups": len(group_ids),
            "images": sum(len(pairs_by_group[gid]) for gid in group_ids),
        }
        for class_id in range(1, 6):
            row[f"{CLASS_NAMES_BY_ID[class_id]}_pixels"] = int(pixels[class_id - 1])
        summary_rows.append(row)

    with (args.output_dir / "split_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(json.dumps({name: len(ids) for name, ids in splits.items()}, indent=2))
    print(f"Prepared dataset at {args.output_dir}")


if __name__ == "__main__":
    main()
