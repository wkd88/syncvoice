#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors:  Han Zhu,
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

"""Compute LibriSpeech WER with a locally downloaded HuBERT model."""

import argparse
import logging
import re
from pathlib import Path


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute WER with HuBERT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--wav-path", type=Path, required=True)
    parser.add_argument("--extension", default="wav")
    parser.add_argument("--decode-path", type=Path)
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="TTS_eval_models checkpoint directory.",
    )
    parser.add_argument(
        "--test-list", type=Path, default=Path("transcript.tsv")
    )
    parser.add_argument("--batch-size", type=int, default=16)
    return parser


def post_process(text: str) -> str:
    text = text.replace("‘", "'").replace("’", "'")
    text = re.sub(r"[^a-zA-Z0-9']", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def process_one(hypothesis: str, truth: str):
    from jiwer import process_words

    truth = post_process(truth)
    hypothesis = post_process(hypothesis)
    measures = process_words(truth, hypothesis)
    return (
        truth,
        hypothesis,
        measures.wer,
        measures.substitutions,
        measures.deletions,
        measures.insertions,
        len(truth.split()),
    )


class SpeechEvalDataset:
    """Lazy audio dataset compatible with the ASR pipeline."""

    def __init__(
        self, wav_path: str, test_list: str, extension: str = "wav"
    ):
        self.wav_names = []
        self.wav_paths = []
        self.transcripts = []
        with Path(test_list).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 2:
                    raise ValueError(
                        f"Invalid transcript line {line_number}"
                    )
                self.wav_names.append(fields[0])
                self.wav_paths.append(
                    Path(wav_path)
                    / (fields[0] + "." + extension.lstrip("."))
                )
                self.transcripts.append(fields[-1])

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, index):
        from zipvoice.eval.utils import load_waveform

        return {
            "array": load_waveform(
                str(self.wav_paths[index]), 16000, return_numpy=True
            ),
            "sampling_rate": 16000,
            "reference": self.transcripts[index],
            "wav_name": self.wav_names[index],
        }


def main() -> int:
    args = get_parser().parse_args()

    import torch
    from tqdm import tqdm
    from transformers import pipeline

    from zipvoice.eval.utils import load_waveform

    model_path = args.model_dir / "wer" / "hubert-large-ls960-ft"
    if not model_path.is_dir():
        raise FileNotFoundError(f"HuBERT model not found: {model_path}")
    if not args.test_list.is_file():
        raise FileNotFoundError(f"Test list not found: {args.test_list}")

    records = []
    with args.test_list.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 2:
                raise ValueError(f"Invalid transcript line {line_number}")
            audio_path = args.wav_path / (
                fields[0] + "." + args.extension.lstrip(".")
            )
            if not audio_path.is_file():
                raise FileNotFoundError(f"Audio not found: {audio_path}")
            records.append(
                {
                    "array": load_waveform(
                        str(audio_path), 16000, return_numpy=True
                    ),
                    "sampling_rate": 16000,
                    "reference": fields[-1],
                    "wav_name": fields[0],
                }
            )
    if not records:
        raise ValueError("No evaluation utterances found")

    device = 0 if torch.cuda.is_available() else -1
    recognizer = pipeline(
        "automatic-speech-recognition",
        model=str(model_path),
        tokenizer=str(model_path),
        device=device,
    )
    results = recognizer(records, batch_size=args.batch_size)

    output = None
    if args.decode_path:
        args.decode_path.parent.mkdir(parents=True, exist_ok=True)
        output = args.decode_path.open("w", encoding="utf-8")
        output.write(
            "Name\tWER\tTruth\tHypothesis\tInsertions\t"
            "Deletions\tSubstitutions\n"
        )

    substitutions = deletions = insertions = word_count = 0
    try:
        for record, result in tqdm(
            zip(records, results), total=len(records)
        ):
            truth, hypothesis, wer, subs, dels, ins, words = process_one(
                result["text"].strip(), record["reference"].strip()
            )
            substitutions += subs
            deletions += dels
            insertions += ins
            word_count += words
            if output:
                output.write(
                    f"{record['wav_name']}\t{wer}\t{truth}\t"
                    f"{hypothesis}\t{ins}\t{dels}\t{subs}\n"
                )
    finally:
        if output:
            output.close()

    wer = round(
        (substitutions + deletions + insertions) / word_count * 100, 2
    )
    logging.info("WER: %.2f%%", wer)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
