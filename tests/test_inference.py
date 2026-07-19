import numpy as np
import torch
from torch import nn

from jaksuhealth_ai.inference import predict_array


class ConstantLesionModel(nn.Module):
    def forward(self, inputs):
        batch, _, height, width = inputs.shape
        logits = torch.zeros(batch, 6, height, width, device=inputs.device)
        logits[:, 1] = 5.0
        return logits


def test_predict_array_restores_original_dimensions():
    image = np.zeros((100, 140), dtype=np.uint8)
    image[30:70, :] = 180
    model = ConstantLesionModel()
    result = predict_array(
        model=model,
        image=image,
        device=torch.device("cpu"),
        output_size=(64, 96),
        use_tta=False,
    )
    assert result["mask"].shape == image.shape
    assert "SRF" in result["lesion_statistics"]
    assert result["lesion_statistics"]["SRF"]["pixels"] > 0
