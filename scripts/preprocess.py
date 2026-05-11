"""Backward-compatible exports. Prefer `from src.preprocess import ...`."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.preprocess import (  # noqa: E402
    augment_image,
    build_datasets,
    create_tf_dataset,
    crop_and_resize,
)
