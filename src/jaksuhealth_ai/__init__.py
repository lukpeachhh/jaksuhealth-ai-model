"""JaksuHealth OCT lesion segmentation package."""

from .constants import CLASS_NAMES, CLASS_NAMES_BY_ID, NUM_CLASSES
from .models import build_model

__all__ = ["CLASS_NAMES", "CLASS_NAMES_BY_ID", "NUM_CLASSES", "build_model"]
__version__ = "1.0.0"
