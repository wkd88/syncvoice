#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors:  Han Zhu
#                                                   Wei Kang)
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

"""Compute SIM-o with a WavLM-based ECAPA-TDNN model."""

import argparse
import logging
from pathlib import Path, PurePosixPath


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate speaker similarity (SIM-o).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--wav-path", type=Path, required=True)
    parser.add_argument("--test-list", type=Path, required=True)
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="TTS_eval_models checkpoint directory.",
    )
    parser.add_argument("--extension", default="wav")
    return parser


def _evaluated_name(reference: str, index=None) -> str:
    reference_path = PurePosixPath(reference.replace("\\", "/"))
    stem = f"{reference_path.parent.name}_{reference_path.stem}"
    return f"{stem}_{index}" if index is not None else stem


class SpeakerSimilarity:
    """WavLM ECAPA-TDNN speaker-similarity evaluator."""

    def __init__(self, sv_model_path: str, ssl_model_path: str):
        import torch

        from zipvoice.eval.models.ecapa_tdnn_wavlm import (
            ECAPA_TDNN_WAVLM,
        )

        self.torch = torch
        self.sample_rate = 16000
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = ECAPA_TDNN_WAVLM(
            feat_dim=1024,
            channels=512,
            emb_dim=256,
            sr=self.sample_rate,
            ssl_model_path=ssl_model_path,
        )
        state_dict = torch.load(sv_model_path, map_location="cpu")
        self.model.load_state_dict(state_dict["model"], strict=False)
        self.model.to(self.device).eval()

    def get_embeddings(self, wav_paths):
        from tqdm import tqdm

        from zipvoice.eval.utils import load_waveform

        embeddings = []
        with self.torch.no_grad():
            for wav_path in tqdm(
                wav_paths, desc="Extracting speaker embeddings"
            ):
                speech = load_waveform(
                    str(wav_path),
                    self.sample_rate,
                    device=self.device,
                    max_seconds=120,
                )
                embeddings.append(self.model([speech]))
        return embeddings

    def score_pairs(self, pairs) -> float:
        import numpy as np

        prompts = self.get_embeddings([pair[0] for pair in pairs])
        evaluated = self.get_embeddings([pair[1] for pair in pairs])
        scores = [
            self.torch.nn.functional.cosine_similarity(
                prompt, output, dim=-1
            ).item()
            for prompt, output in zip(prompts, evaluated)
        ]
        return float(np.mean(scores))

    def score(self, wav_path: str, extension: str, test_list: str) -> float:
        pairs = []
        output_dir = Path(wav_path)
        with Path(test_list).open(encoding="utf-8") as stream:
            for line_number, raw_line in enumerate(stream, 1):
                fields = raw_line.rstrip("\n").split("\t")
                if fields == [""]:
                    continue
                if len(fields) not in (4, 6):
                    raise ValueError(
                        f"Invalid test-list line {line_number}"
                    )
                prompt = Path(fields[0].replace(".mp4", ".wav"))
                indices = range(3) if len(fields) == 6 else (None,)
                for index in indices:
                    if index is not None and not fields[-3 + index]:
                        continue
                    evaluated = output_dir / (
                        _evaluated_name(fields[2], index)
                        + "."
                        + extension.lstrip(".")
                    )
                    if not prompt.is_file() or not evaluated.is_file():
                        missing = (
                            prompt if not prompt.is_file() else evaluated
                        )
                        raise FileNotFoundError(
                            f"Audio not found: {missing}"
                        )
                    pairs.append((prompt, evaluated))
        if not pairs:
            raise ValueError("No evaluation pairs found")
        return self.score_pairs(pairs)


def main() -> int:
    args = get_parser().parse_args()

    checkpoint = (
        args.model_dir
        / "speaker_similarity"
        / "wavlm_large_finetune.pth"
    )
    ssl_dir = args.model_dir / "speaker_similarity" / "wavlm_large"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Speaker checkpoint not found: {checkpoint}")
    if not ssl_dir.is_dir():
        raise FileNotFoundError(f"WavLM directory not found: {ssl_dir}")
    if not args.test_list.is_file():
        raise FileNotFoundError(f"Test list not found: {args.test_list}")

    pairs = []
    with args.test_list.open(encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            fields = raw_line.rstrip("\n").split("\t")
            if fields == [""]:
                continue
            if len(fields) not in (4, 6):
                raise ValueError(
                    f"Invalid test-list line {line_number}: "
                    f"expected 4 or 6 fields"
                )
            prompt = Path(fields[0].replace(".mp4", ".wav"))
            if len(fields) == 6:
                entries = (
                    (_evaluated_name(fields[2], index), text)
                    for index, text in enumerate(fields[-3:])
                    if text
                )
            else:
                entries = [(_evaluated_name(fields[2]), fields[-1])]
            for name, _ in entries:
                evaluated = args.wav_path / (
                    name + "." + args.extension.lstrip(".")
                )
                if not prompt.is_file():
                    raise FileNotFoundError(f"Prompt not found: {prompt}")
                if not evaluated.is_file():
                    raise FileNotFoundError(
                        f"Evaluated audio not found: {evaluated}"
                    )
                pairs.append((prompt, evaluated))
    if not pairs:
        raise ValueError("No evaluation pairs found")

    evaluator = SpeakerSimilarity(str(checkpoint), str(ssl_dir))
    logging.info("SIM-o score: %.3f", evaluator.score_pairs(pairs))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
