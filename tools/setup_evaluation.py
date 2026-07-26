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

"""Install the optional evaluation backend and model checkpoints."""

import argparse
import http.client
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LATENTSYNC_DIR = ROOT / "sup" / "LatentSync"
LATENTSYNC_URL = "https://github.com/bytedance/LatentSync.git"
LATENTSYNC_ARCHIVE = (
    "https://github.com/bytedance/LatentSync/archive/refs/heads/main.zip"
)
LATENTSYNC_REPO = "ByteDance/LatentSync-1.6"
TTS_REPO = "k2-fsa/TTS_eval_models"
LATENTSYNC_FILES = (
    "auxiliary/syncnet_v2.model",
    "auxiliary/sfd_face.pth",
)
TTS_PATTERNS = (
    "speaker_similarity/**",
    "mos/**",
    "wer/**",
)
TTS_REQUIRED_FILES = (
    "speaker_similarity/wavlm_large_finetune.pth",
    "speaker_similarity/wavlm_large/hubconf.py",
    "speaker_similarity/wavlm_large/wavlm_large.pt",
    "mos/utmos22_strong_step7459_v1.pt",
    "wer/hubert-large-ls960-ft/pytorch_model.bin",
    "wer/paraformer-zh/model.pt",
    "wer/whisper-large-v3/pytorch_model.bin",
)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Clone ByteDance LatentSync and download optional evaluation "
            "checkpoints from Hugging Face."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--with-tts-models",
        action="store_true",
        help="Also download speaker similarity, MOS, and WER models.",
    )
    parser.add_argument(
        "--revision",
        help="Hugging Face repository revision (branch, tag, or commit).",
    )
    return parser


def _clone_latentsync() -> None:
    if LATENTSYNC_DIR.exists():
        if (LATENTSYNC_DIR / "eval" / "eval_sync_conf.py").is_file():
            return
        if not any(LATENTSYNC_DIR.iterdir()):
            LATENTSYNC_DIR.rmdir()
        else:
            raise RuntimeError(
                f"{LATENTSYNC_DIR} exists but is not a Git checkout"
            )
    parent = LATENTSYNC_DIR.parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent) as temporary:
        temporary_dir = Path(temporary)
        checkout = temporary_dir / "LatentSync"
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    LATENTSYNC_URL,
                    str(checkout),
                ],
                cwd=temporary_dir,
                check=True,
            )
        except subprocess.CalledProcessError:
            archive = temporary_dir / "LatentSync-main.zip"
            try:
                urllib.request.urlretrieve(LATENTSYNC_ARCHIVE, archive)
            except (
                http.client.IncompleteRead,
                urllib.error.URLError,
            ) as error:
                raise RuntimeError(
                    "Unable to download LatentSync from GitHub. Clone "
                    f"{LATENTSYNC_URL} manually into {LATENTSYNC_DIR}, "
                    "then run this command again."
                ) from error
            with zipfile.ZipFile(archive) as source:
                source.extractall(temporary_dir)
            checkout = temporary_dir / "LatentSync-main"
        shutil.move(str(checkout), LATENTSYNC_DIR)


def _download(
    repo_id: str,
    destination: Path,
    allow_patterns,
    revision: str | None,
) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface_hub is required; install requirements.txt first"
        ) from error
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=destination,
        allow_patterns=list(allow_patterns),
    )


def _require_files(root: Path, relative_paths) -> None:
    missing = [
        str(root / relative)
        for relative in relative_paths
        if not (root / relative).is_file()
    ]
    if missing:
        raise RuntimeError(
            "Download completed without required files: "
            + ", ".join(missing)
        )


def main() -> int:
    args = get_parser().parse_args()
    _clone_latentsync()

    latent_checkpoints = LATENTSYNC_DIR / "checkpoints"
    _download(
        LATENTSYNC_REPO,
        latent_checkpoints,
        LATENTSYNC_FILES,
        args.revision,
    )
    _require_files(latent_checkpoints, LATENTSYNC_FILES)

    if args.with_tts_models:
        tts_dir = ROOT / "checkpoints" / "evaluation" / "tts"
        _download(TTS_REPO, tts_dir, TTS_PATTERNS, args.revision)
        _require_files(tts_dir, TTS_REQUIRED_FILES)

    print("Evaluation dependencies are ready.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
