"""
Root training entry point (GitHub layout).

Equivalent to: python -m src.train [arguments...]
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train import main

if __name__ == "__main__":
    main()
