#!/usr/bin/env python3
"""Clone visual backends and collect extractor weights under checkpoints/visual/."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUP = ROOT / "sup"
VISUAL_DIR = ROOT / "checkpoints" / "visual"

REPOSITORIES = (
    (
        "Lipreading_using_Temporal_Convolutional_Networks",
        "https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks.git",
        True,
    ),
    (
        "emonet",
        "https://github.com/face-analysis/emonet.git",
        False,
    ),
    (
        "Lipreading_using_Temporal_Convolutional_Networks/face_detection",
        "https://github.com/hhj1897/face_detection.git",
        False,
    ),
    (
        "Lipreading_using_Temporal_Convolutional_Networks/face_alignment",
        "https://github.com/hhj1897/face_alignment.git",
        False,
    ),
)

# Local copies after Git LFS pull (or after clone for non-LFS files).
LOCAL_WEIGHTS = (
    (
        "Resnet50_Final.pth",
        SUP
        / "Lipreading_using_Temporal_Convolutional_Networks"
        / "face_detection"
        / "ibug"
        / "face_detection"
        / "retina_face"
        / "weights"
        / "Resnet50_Final.pth",
    ),
    (
        "2dfan2.pth",
        SUP
        / "Lipreading_using_Temporal_Convolutional_Networks"
        / "face_alignment"
        / "ibug"
        / "face_alignment"
        / "fan"
        / "weights"
        / "2dfan2.pth",
    ),
    (
        "emonet_8.pth",
        SUP / "emonet" / "pretrained" / "emonet_8.pth",
    ),
)

# Upstream Lipreading TCN model-zoo entry: resnet18_dctcn_video
LIP_GDRIVE_ID = "179NgMsHo9TeZCLLtNWFVgRehDvzteMZE"
LIP_WEIGHT_NAME = "lrw_resnet18_dctcn_video.pth"
MIN_WEIGHT_BYTES = 1_000_000


def _run(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True)


def _is_real_weight(path: Path) -> bool:
    return path.is_file() and path.stat().st_size >= MIN_WEIGHT_BYTES


def _clone_or_update(name: str, url: str, recursive: bool) -> Path:
    destination = SUP / name
    destination.mkdir(parents=True, exist_ok=True)
    if not (destination / ".git").exists():
        _run("init", cwd=destination)
        _run("remote", "add", "upstream", url, cwd=destination)
    _run("fetch", "--depth", "1", "upstream", "HEAD", cwd=destination)
    _run("checkout", "FETCH_HEAD", "--", ".", cwd=destination)
    if recursive and (destination / ".gitmodules").is_file():
        _run("submodule", "update", "--init", "--recursive", cwd=destination)
    return destination


def _pull_lfs(repo: Path) -> None:
    try:
        subprocess.run(
            ["git", "lfs", "install", "--local"],
            cwd=repo,
            check=False,
            capture_output=True,
        )
        result = subprocess.run(
            ["git", "lfs", "pull"],
            cwd=repo,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(
                f"Warning: git lfs pull failed in {repo} "
                f"(install Git LFS if weights look incomplete)."
            )
    except FileNotFoundError:
        print("Warning: git is unavailable; cannot pull LFS weights.")


def _copy_weight(name: str, source: Path) -> None:
    destination = VISUAL_DIR / name
    if _is_real_weight(destination):
        print(f"OK  {destination}")
        return
    if not _is_real_weight(source):
        raise SystemExit(
            f"Missing or incomplete weight: {source}\n"
            "Install Git LFS (https://git-lfs.github.com/), then rerun:\n"
            "  python tools/setup_visual_backends.py"
        )
    shutil.copy2(source, destination)
    print(f"OK  {destination}")


def _download_lip_weight() -> None:
    destination = VISUAL_DIR / LIP_WEIGHT_NAME
    if _is_real_weight(destination):
        print(f"OK  {destination}")
        return

    try:
        import gdown
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "gdown"],
            check=True,
        )
        import gdown

    print(f"Downloading {LIP_WEIGHT_NAME} from Google Drive...")
    gdown.download(id=LIP_GDRIVE_ID, output=str(destination), quiet=False)
    if not _is_real_weight(destination):
        raise SystemExit(
            f"Failed to download {destination}. Get resnet18_dctcn_video from "
            "https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks#model-zoo "
            f"and save it as {destination}."
        )
    print(f"OK  {destination}")


def main() -> None:
    SUP.mkdir(parents=True, exist_ok=True)
    VISUAL_DIR.mkdir(parents=True, exist_ok=True)

    for name, url, recursive in REPOSITORIES:
        repo = _clone_or_update(name, url, recursive)
        _pull_lfs(repo)

    for name, source in LOCAL_WEIGHTS:
        _copy_weight(name, source)
    _download_lip_weight()

    print(f"Visual backends and weights are ready under: {VISUAL_DIR}")


if __name__ == "__main__":
    main()
