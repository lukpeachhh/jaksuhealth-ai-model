# Model pipeline

## 1. Input

- Single grayscale OCT B-scan.
- Integer mask with the same filename and spatial dimensions during training.
- Class IDs are defined in `jaksuhealth_ai.constants`.

## 2. Grouped split

The preparation script groups images by the filename prefix before `_`. For
AMD-SD filenames (`n_x.png`), this prefix is the eye ID. An eye can belong to
only one of train, validation, or test. The script uses a deterministic greedy
assignment to approximate the requested group counts while balancing lesion
pixels. A true patient-level split requires the eye-to-patient mapping from the
original dataset metadata.

## 3. Retina ROI preprocessing

1. Gaussian blur.
2. Otsu thresholding.
3. Morphological closing.
4. Largest-contour detection.
5. Vertical crop with a configurable margin.
6. Resize to 384 × 576 by default.

Masks use nearest-neighbor interpolation. If no contour is found, the full image
is resized as a fallback.

## 4. Augmentation

The training pipeline applies horizontal flip, affine transformation, mild
non-rigid distortion, Gaussian noise, brightness/contrast adjustment, and
CLAHE. Validation and test inputs are deterministic.

## 5. Model comparison

U-Net, U-Net++, and DeepLabV3+ are created by one model factory. Their default
configs use the same EfficientNet-B3 encoder and six output classes.

## 6. Loss

The training objective combines:

- class-weighted focal cross-entropy;
- multiclass Tversky loss over foreground classes.

This is intended to reduce the effect of background dominance and improve
sensitivity to underrepresented lesions.

## 7. Selection and evaluation

The best checkpoint is selected by validation macro IoU excluding background.
Final test evaluation reports macro/micro metrics, per-class metrics, parameter
count, and average inference time per image.
