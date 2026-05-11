"""Backward-compatible exports. Prefer `from src.data_loader import ...`."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_loader import (  # noqa: E402
    CLASS_NAMES,
    DATA_ROOT,
    NUM_CLASSES,
    PROJECT_ROOT,
    SPLITS,
    build_annotation_manifest,
    get_class_counts,
    get_image_path_for_label,
    load_class_names,
    parse_label_line,
    save_manifest_csv,
)

ROOT = PROJECT_ROOT
