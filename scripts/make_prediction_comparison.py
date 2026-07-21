from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


CLASS_COLORS = {
    0: (0, 0, 0),          # Background
    1: (255, 0, 0),        # SRF
    2: (0, 255, 0),        # IRF
    3: (0, 0, 255),        # PED
    4: (255, 255, 0),      # SHRM
    5: (255, 105, 180),    # IS/OS
}

CLASS_NAMES = {
    0: "Background",
    1: "SRF",
    2: "IRF",
    3: "PED",
    4: "SHRM",
    5: "IS/OS",
}


def read_grayscale_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    return mask


def apply_colormap_to_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in CLASS_COLORS.items():
        color_mask[mask == class_id] = color

    return color_mask


def overlay_mask_on_grayscale(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    base = np.stack([image, image, image], axis=-1).astype(np.uint8)
    color_mask = apply_colormap_to_mask(mask)

    overlay = base.copy()
    foreground = mask > 0

    overlay[foreground] = (
        (1.0 - alpha) * base[foreground] + alpha * color_mask[foreground]
    ).astype(np.uint8)

    return overlay


def build_legend_text() -> str:
    parts = []
    for class_id in range(1, 6):
        parts.append(f"{class_id}={CLASS_NAMES[class_id]}")
    return " | ".join(parts)


def find_matching_files(
    image_dir: Path,
    gt_mask_dir: Path,
    unet_dir: Path,
    unetpp_dir: Path,
    deeplab_dir: Path,
) -> list[str]:
    image_names = {p.name for p in image_dir.glob("*.png")}
    gt_names = {p.name for p in gt_mask_dir.glob("*.png")}
    unet_names = {p.name for p in unet_dir.glob("*.png")}
    unetpp_names = {p.name for p in unetpp_dir.glob("*.png")}
    deeplab_names = {p.name for p in deeplab_dir.glob("*.png")}

    common = (
        image_names
        & gt_names
        & unet_names
        & unetpp_names
        & deeplab_names
    )

    return sorted(common)


def create_comparison_figure(
    filenames: list[str],
    image_dir: Path,
    gt_mask_dir: Path,
    unet_dir: Path,
    unetpp_dir: Path,
    deeplab_dir: Path,
    output_path: Path,
) -> None:
    if len(filenames) == 0:
        raise ValueError("No matching files were found.")

    n_rows = len(filenames)
    n_cols = 5

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(20, 4 * n_rows),
    )

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    column_titles = [
        "Original OCT",
        "Ground Truth",
        "U-Net",
        "U-Net++",
        "DeepLabV3+",
    ]

    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=14, fontweight="bold")

    for row, filename in enumerate(filenames):
        image = read_grayscale_image(image_dir / filename)

        gt_mask = read_mask(gt_mask_dir / filename)
        unet_mask = read_mask(unet_dir / filename)
        unetpp_mask = read_mask(unetpp_dir / filename)
        deeplab_mask = read_mask(deeplab_dir / filename)

        gt_overlay = overlay_mask_on_grayscale(image, gt_mask)
        unet_overlay = overlay_mask_on_grayscale(image, unet_mask)
        unetpp_overlay = overlay_mask_on_grayscale(image, unetpp_mask)
        deeplab_overlay = overlay_mask_on_grayscale(image, deeplab_mask)

        panels = [
            image,
            gt_overlay,
            unet_overlay,
            unetpp_overlay,
            deeplab_overlay,
        ]

        for col, panel in enumerate(panels):
            ax = axes[row, col]

            if col == 0:
                ax.imshow(panel, cmap="gray")
            else:
                ax.imshow(panel)

            ax.axis("off")

            if col == 0:
                ax.set_ylabel(
                    filename,
                    fontsize=11,
                    rotation=90,
                    labelpad=12,
                )

    legend_text = build_legend_text()

    fig.suptitle(
        "Prediction Comparison on 5 Test Samples",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )

    fig.text(
        0.5,
        0.01,
        f"Overlay colors: {legend_text}",
        ha="center",
        fontsize=11,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a 5-sample prediction comparison figure."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing original grayscale OCT images.",
    )
    parser.add_argument(
        "--gt-mask-dir",
        type=Path,
        required=True,
        help="Directory containing ground-truth masks.",
    )
    parser.add_argument(
        "--unet-dir",
        type=Path,
        required=True,
        help="Directory containing U-Net predicted masks.",
    )
    parser.add_argument(
        "--unetpp-dir",
        type=Path,
        required=True,
        help="Directory containing U-Net++ predicted masks.",
    )
    parser.add_argument(
        "--deeplab-dir",
        type=Path,
        required=True,
        help="Directory containing DeepLabV3+ predicted masks.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("results/figures/prediction_comparison_5_samples.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to visualize.",
    )

    args = parser.parse_args()

    matched_files = find_matching_files(
        image_dir=args.image_dir,
        gt_mask_dir=args.gt_mask_dir,
        unet_dir=args.unet_dir,
        unetpp_dir=args.unetpp_dir,
        deeplab_dir=args.deeplab_dir,
    )

    if len(matched_files) < args.num_samples:
        raise ValueError(
            f"Only found {len(matched_files)} matched files, "
            f"but requested {args.num_samples}."
        )

    selected_files = matched_files[: args.num_samples]

    print("Selected files:")
    for name in selected_files:
        print("-", name)

    create_comparison_figure(
        filenames=selected_files,
        image_dir=args.image_dir,
        gt_mask_dir=args.gt_mask_dir,
        unet_dir=args.unet_dir,
        unetpp_dir=args.unetpp_dir,
        deeplab_dir=args.deeplab_dir,
        output_path=args.output_path,
    )

    print(f"Saved figure to: {args.output_path}")


if __name__ == "__main__":
    main()