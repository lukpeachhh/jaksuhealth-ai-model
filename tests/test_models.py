import pytest
import torch

smp = pytest.importorskip("segmentation_models_pytorch")

torch.set_num_threads(1)

from jaksuhealth_ai.models import build_model


@pytest.mark.parametrize(
    "architecture", ["unet", "unetplusplus", "deeplabv3plus"]
)
def test_model_output_shape(architecture):
    model = build_model(
        architecture=architecture,
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=1,
        num_classes=6,
    )
    model.eval()
    inputs = torch.randn(1, 1, 64, 64)
    with torch.inference_mode():
        outputs = model(inputs)
    assert outputs.shape == (1, 6, 64, 64)
