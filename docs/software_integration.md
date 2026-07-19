# Software integration contract

This document describes the AI component used by the complete JaksuHealth
software. The application code itself is outside the scope of this repository.

## Input

| Field | Type | Description |
|---|---|---|
| OCT image | PNG/JPEG/TIFF or NumPy array | Grayscale B-scan |
| Model checkpoint | `.pth` | Checkpoint produced by `scripts/train.py` |
| Config | YAML | Architecture and preprocessing parameters |

## Output

The inference pipeline returns:

```python
{
    "mask": np.ndarray,                # original image resolution
    "processed_mask": np.ndarray,      # model input resolution
    "processed_image": np.ndarray,
    "metadata": CropMetadata,
    "lesion_statistics": {
        "SRF": {"pixels": int, "percentage": float},
        "IRF": {"pixels": int, "percentage": float},
        "PED": {"pixels": int, "percentage": float},
        "SHRM": {"pixels": int, "percentage": float},
        "IS/OS": {"pixels": int, "percentage": float},
    },
}
```

## Example

```python
import torch

from jaksuhealth_ai.inference import predict_file
from jaksuhealth_ai.models import build_model, load_checkpoint

model = build_model(
    architecture="unet",
    encoder_name="efficientnet-b3",
    encoder_weights=None,
    in_channels=1,
    num_classes=6,
)
load_checkpoint(model, "best_model.pth")

result = predict_file(
    model=model,
    image_path="example_oct.png",
    device=torch.device("cpu"),
)
```

## Safety

The output is a model-generated segmentation and must be presented as decision
support, not as an autonomous diagnosis. Production integration should include
input validation, audit logs, versioned checkpoints, uncertainty handling, and
clinical review.
