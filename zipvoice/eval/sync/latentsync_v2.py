#!/usr/bin/env python3
# Copyright 2026 SyncVoice contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run ByteDance LatentSync's official v2 sync-confidence evaluator.

This wrapper intentionally delegates evaluation to the official open-source
``eval.eval_sync_conf`` module and requires the v2 SyncNet checkpoint. It does
not contain or invoke the legacy ZipVoice sync evaluator.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BACKEND = PROJECT_ROOT / "sup" / "LatentSync"
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate sync confidence and AV offset with ByteDance "
            "LatentSync's official eval module and v2 checkpoint."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--video", type=Path, help="Single video to evaluate.")
    inputs.add_argument(
        "--input-dir",
        type=Path,
        help="Directory of videos to evaluate one by one.",
    )
    parser.add_argument(
        "--backend-dir",
        type=Path,
        default=DEFAULT_BACKEND,
        help="ByteDance LatentSync checkout.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "SyncNet v2 checkpoint; defaults to "
            "BACKEND/checkpoints/auxiliary/syncnet_v2.model."
        ),
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        help="Temporary directory used by the official evaluator.",
    )
    return parser


def _validate(args) -> tuple[Path, Path, Path]:
    backend = args.backend_dir.expanduser().resolve()
    if not backend.is_dir():
        raise FileNotFoundError(f"LatentSync backend not found: {backend}")
    evaluator = backend / "eval" / "eval_sync_conf.py"
    if not evaluator.is_file():
        raise FileNotFoundError(
            f"Official LatentSync evaluator not found: {evaluator}"
        )

    checkpoint = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint
        else backend / "checkpoints" / "auxiliary" / "syncnet_v2.model"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"SyncNet v2 checkpoint not found: {checkpoint}"
        )
    if checkpoint.name != "syncnet_v2.model":
        raise ValueError(
            "--checkpoint must point to the LatentSync v2 checkpoint "
            "named syncnet_v2.model"
        )

    face_checkpoint = (
        backend / "checkpoints" / "auxiliary" / "sfd_face.pth"
    )
    if not face_checkpoint.is_file():
        raise FileNotFoundError(
            f"S3FD face-detector checkpoint not found: {face_checkpoint}"
        )

    temp_dir = (
        args.temp_dir.expanduser().resolve()
        if args.temp_dir
        else backend / "temp" / "sync-eval"
    )
    temp_dir.mkdir(parents=True, exist_ok=True)
    return backend, checkpoint, temp_dir


def _videos(args) -> Iterable[Path]:
    if args.video:
        video = args.video.expanduser().resolve()
        if not video.is_file():
            raise FileNotFoundError(f"Input video not found: {video}")
        yield video
        return

    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    videos = sorted(
        path.resolve()
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    if not videos:
        raise ValueError(f"No supported videos found in: {input_dir}")
    yield from videos


def _run_one(
    video: Path, backend: Path, checkpoint: Path, temp_dir: Path
) -> int:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (str(backend), existing_pythonpath)
        if part
    )
    command = [
        sys.executable,
        "-m",
        "eval.eval_sync_conf",
        "--initial_model",
        str(checkpoint),
        "--video_path",
        str(video),
        "--temp_dir",
        str(temp_dir),
    ]
    print(f"Evaluating: {video}", flush=True)
    completed = subprocess.run(command, cwd=backend, env=env, check=False)
    return completed.returncode


def main() -> int:
    args = get_parser().parse_args()
    backend, checkpoint, temp_dir = _validate(args)
    failed = []
    for video in _videos(args):
        if _run_one(video, backend, checkpoint, temp_dir) != 0:
            failed.append(video)
    if failed:
        print(
            "LatentSync evaluation failed for: "
            + ", ".join(str(path) for path in failed),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
