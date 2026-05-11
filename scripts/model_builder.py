"""Backward-compatible exports. Prefer `from src.model import ...`."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.model import build_scratch_cnn, build_transfer_model  # noqa: E402
