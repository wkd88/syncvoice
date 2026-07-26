#!/usr/bin/env python3
"""Download and validate the SyncVoice acoustic model from Hugging Face."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

EXPECTED_FILES = (
    "lrs_exp/checkpoint-55000.pt",
    "lrs_exp/model.json",
    "lrs_exp/tokens.txt",
)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download the SyncVoice acoustic model."
    )
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("SYNCVOICE_REPO_ID", "kaidi555/SyncVoice"),
        help="Hugging Face model repository, or set SYNCVOICE_REPO_ID",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/syncvoice"),
    )
    parser.add_argument("--revision", default="main")
    return parser


def main() -> None:
    args = get_parser().parse_args()
    if not args.repo_id:
        raise SystemExit(
            "Provide --repo-id OWNER/REPOSITORY or set SYNCVOICE_REPO_ID."
        )
    snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=args.output_dir,
        allow_patterns=list(EXPECTED_FILES),
    )
    missing = [
        relative
        for relative in EXPECTED_FILES
        if not (args.output_dir / relative).is_file()
    ]
    if missing:
        raise SystemExit("Missing downloaded files:\n- " + "\n- ".join(missing))
    print(f"Weights are ready under: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
