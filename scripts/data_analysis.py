"""Backward-compatible entry: python scripts/data_analysis.py"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_analysis import main  # noqa: E402

if __name__ == "__main__":
    main()
