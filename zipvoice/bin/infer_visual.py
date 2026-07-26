#!/usr/bin/env python3
"""Backward-compatible CLI for visual feature extraction.

Prefer ``tools/extract_visual_features.py`` for new scripts.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    tool = Path(__file__).resolve().parents[2] / "tools" / "extract_visual_features.py"
    sys.argv[0] = str(tool)
    runpy.run_path(str(tool), run_name="__main__")


if __name__ == "__main__":
    main()
