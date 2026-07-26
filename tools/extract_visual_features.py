#!/usr/bin/env python3
"""Extract SyncVoice lip and face conditioning features."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract 512-D lip and 256-D face features from MP4 videos."
    )
    parser.add_argument("input", type=Path, help="An MP4 file or a directory")
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=None,
        help="Directory containing the four visual model weight files",
    )
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu")
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = get_parser().parse_args()
    from zipvoice.visual.extractor import VisualFeatureExtractor, iter_videos

    videos = list(iter_videos(args.input, recursive=not args.no_recursive))
    if not videos:
        raise SystemExit(f"No MP4 videos found under: {args.input}")

    extractor = VisualFeatureExtractor(
        weights_dir=args.weights_dir, device=args.device
    )
    failures = []
    for video in tqdm(videos, desc="Extracting visual features"):
        try:
            extractor.extract_to_files(video, overwrite=args.overwrite)
        except Exception:
            logging.exception("Failed to process %s", video)
            failures.append(video)

    if failures:
        raise SystemExit(f"Feature extraction failed for {len(failures)} video(s).")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    main()
