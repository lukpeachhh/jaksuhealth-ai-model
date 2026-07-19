"""PyTorch dataset and Albumentations pipelines for AMD OCT segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import NUM_CLASSES, SUPPORTED_IMAGE_EXTENSIONS
from .preprocessing import (
    crop_retina_roi,
    normalize_grayscale,
    read_grayscale,
    resize_image_and_mask,
    validate_mask_labels,
)


def build_train_transform() -> A.Compose:
    """Create the training augmentation pipeline using Albumentations 2.x API."""

    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.0625, 0.0625),
                rotate=(-15, 15),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
                p=0.5,
            ),
            A.OneOf(
                [
                    A.GridDistortion(
                        num_steps=5,
                        distort_limit=0.05,
                        border_mode=cv2.BORDER_CONSTANT,
                        fill=0,
                        fill_mask=0,
                        p=1.0,
                    ),
                    A.OpticalDistortion(
                        distort_limit=(-0.05, 0.05),
                        border_mode=cv2.BORDER_CONSTANT,
                        fill=0,
                        fill_mask=0,
                        p=1.0,
                    ),
                    A.ElasticTransform(
                        alpha=1.0,
                        sigma=50.0,
                        border_mode=cv2.BORDER_CONSTANT,
                        fill=0,
                        fill_mask=0,
                        p=1.0,
                    ),
                ],
                p=0.3,
            ),
            A.OneOf(
                [
                    A.GaussNoise(std_range=(0.01, 0.03), p=1.0),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=1.0
                    ),
                    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
                ],
                p=0.4,
            ),
        ]
    )


def build_validation_transform() -> A.Compose:
    """Create the deterministic validation/test transform."""

    return A.Compose([])


class OCTSegmentationDataset(Dataset):
    """Dataset for paired OCT images and integer segmentation masks."""

    def __init__(
        self,
        split_dir: str | Path,
        split_name: str,
        transform: Callable | None = None,
        use_retina_crop: bool = True,
        crop_margin: int = 30,
        output_size: tuple[int, int] = (384, 576),
        preload: bool = False,
        validate_masks: bool = True,
        return_metadata: bool = False,
    ) -> None:
        self.split_dir = Path(split_dir)
        self.split_name = split_name
        self.image_dir = self.split_dir / split_name / "images"
        self.mask_dir = self.split_dir / split_name / "masks"
        self.transform = transform
        self.use_retina_crop = use_retina_crop
        self.crop_margin = crop_margin
        self.output_size = output_size
        self.preload = preload
        self.validate_masks = validate_masks
        self.return_metadata = return_metadata

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        self.image_names = sorted(
            path.name
            for path in self.image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )
        if not self.image_names:
            raise ValueError(f"No supported images found in {self.image_dir}")

        missing_masks = [
            name for name in self.image_names if not (self.mask_dir / name).is_file()
        ]
        if missing_masks:
            preview = ", ".join(missing_masks[:5])
            raise FileNotFoundError(
                f"Missing {len(missing_masks)} masks in {self.mask_dir}; "
                f"examples: {preview}"
            )

        self._cache: list[tuple[np.ndarray, np.ndarray]] | None = None
        if self.preload:
            self._cache = [self._read_pair(name) for name in self.image_names]

    def __len__(self) -> int:
        return len(self.image_names)

    def _read_pair(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        image = read_grayscale(self.image_dir / name)
        mask = read_grayscale(self.mask_dir / name)
        if image.shape != mask.shape:
            raise ValueError(
                f"Shape mismatch for {name}: image {image.shape}, mask {mask.shape}"
            )
        if self.validate_masks:
            validate_mask_labels(mask, num_classes=NUM_CLASSES)
        return image, mask

    def __getitem__(self, index: int):
        name = self.image_names[index]
        if self._cache is None:
            image, mask = self._read_pair(name)
        else:
            cached_image, cached_mask = self._cache[index]
            image, mask = cached_image.copy(), cached_mask.copy()

        if self.use_retina_crop:
            image, mask, metadata = crop_retina_roi(
                image,
                mask,
                output_size=self.output_size,
                margin=self.crop_margin,
            )
        else:
            image, mask, metadata = resize_image_and_mask(
                image, mask, self.output_size
            )

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image_tensor = torch.from_numpy(
            normalize_grayscale(image)[None, ...]
        ).float()
        mask_tensor = torch.from_numpy(mask.astype(np.int64)).long()

        if self.return_metadata:
            return {
                "image": image_tensor,
                "mask": mask_tensor,
                "filename": name,
                "metadata": metadata.to_dict(),
            }
        return image_tensor, mask_tensor
