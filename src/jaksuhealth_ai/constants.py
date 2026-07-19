"""Project-wide constants and the canonical lesion label mapping.

The integer mapping below follows the mapping verified for the JaksuHealth
experiments. The Kaggle AMD-SD data card has historically shown IRF and PED in
a different order, so every experiment should record the exact mapping used.
"""

from __future__ import annotations

from typing import Final

CLASS_NAMES_BY_ID: Final[dict[int, str]] = {
    0: "Background",
    1: "SRF",
    2: "IRF",
    3: "PED",
    4: "SHRM",
    5: "IS/OS",
}

CLASS_NAMES: Final[tuple[str, ...]] = tuple(
    CLASS_NAMES_BY_ID[index] for index in sorted(CLASS_NAMES_BY_ID)
)
NUM_CLASSES: Final[int] = len(CLASS_NAMES)
BACKGROUND_CLASS_ID: Final[int] = 0
IGNORE_INDEX: Final[int] = 255

# RGB colors used in exported visualizations.
CLASS_COLORS: Final[tuple[tuple[int, int, int], ...]] = (
    (0, 0, 0),        # Background: black
    (255, 0, 0),      # SRF: red
    (0, 255, 0),      # IRF: green
    (0, 0, 255),      # PED: blue
    (255, 255, 0),    # SHRM: yellow
    (255, 105, 180),  # IS/OS: pink
)

SUPPORTED_IMAGE_EXTENSIONS: Final[tuple[str, ...]] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
)
