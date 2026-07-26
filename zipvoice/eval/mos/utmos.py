#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../../../LICENSE for clarification regarding multiple authors
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

"""Calculate UTMOS scores for generated audio."""

import argparse
import logging
from pathlib import Path


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate UTMOS scores with UTMOS22Strong.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--wav-path",
        type=Path,
        required=True,
        help="Directory containing audio files.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="TTS_eval_models checkpoint directory.",
    )
    parser.add_argument("--extension", default="wav")
    return parser


class UTMOSScore:
    """Predict UTMOS scores for audio clips."""

    def __init__(self, model_path: str):
        import torch

        from zipvoice.eval.models.utmos import UTMOS22Strong

        self.torch = torch
        self.sample_rate = 16000
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = UTMOS22Strong()
        state_dict = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

    def score_files(self, wav_paths):
        from tqdm import tqdm

        from zipvoice.eval.utils import load_waveform

        scores = []
        with self.torch.no_grad():
            for wav_path in tqdm(wav_paths, desc="Scoring audio files"):
                speech = load_waveform(
                    str(wav_path),
                    self.sample_rate,
                    device=self.device,
                )
                score = self.model(
                    speech.unsqueeze(0), self.sample_rate
                )
                scores.append(score.item())
        return scores

    def score_dir(self, dir_path: str, extension: str) -> float:
        import numpy as np

        directory = Path(dir_path)
        suffix = "." + extension.lstrip(".").lower()
        wav_paths = sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() == suffix
        )
        if not wav_paths:
            raise ValueError(f"No {suffix} files found in {directory}")
        return float(np.mean(self.score_files(wav_paths)))


def main() -> int:
    args = get_parser().parse_args()

    if not args.wav_path.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {args.wav_path}")
    model_path = args.model_dir / "mos" / "utmos22_strong_step7459_v1.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"UTMOS checkpoint not found: {model_path}")
    evaluator = UTMOSScore(str(model_path))
    score = evaluator.score_dir(str(args.wav_path), args.extension)
    logging.info("UTMOS score: %.2f", score)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
