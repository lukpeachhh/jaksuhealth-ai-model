import numpy as np

from jaksuhealth_ai.preprocessing import crop_retina_roi, inverse_crop_mask


def test_crop_retina_roi_preserves_mask_labels_and_size():
    image = np.zeros((200, 300), dtype=np.uint8)
    image[70:130, :] = 180
    mask = np.zeros_like(image)
    mask[85:105, 100:180] = 2

    cropped_image, cropped_mask, metadata = crop_retina_roi(
        image, mask, output_size=(96, 128), margin=10
    )

    assert cropped_image.shape == (96, 128)
    assert cropped_mask.shape == (96, 128)
    assert set(np.unique(cropped_mask)).issubset({0, 2})
    assert 0 <= metadata.y_min < metadata.y_max <= image.shape[0]


def test_inverse_crop_mask_returns_original_shape():
    image = np.zeros((120, 160), dtype=np.uint8)
    image[40:80, :] = 200
    cropped_image, _, metadata = crop_retina_roi(
        image, output_size=(64, 96), margin=5
    )
    prediction = np.ones(cropped_image.shape, dtype=np.uint8)
    restored = inverse_crop_mask(prediction, metadata)
    assert restored.shape == image.shape
    assert restored.dtype == np.uint8
