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

"""Compute Seed-TTS WER with Whisper (English) or Paraformer (Chinese)."""

import argparse
import logging
import re
import string
from pathlib import Path, PurePosixPath


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute WER following the Seed-TTS protocol.",
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
    parser.add_argument("--test-list", type=Path, default=Path("test.tsv"))
    parser.add_argument("--lang", choices=("zh", "en"), required=True)
    return parser


def get_suppressed_tokens(tokenizer):
    unwanted = set("0123456789$£")
    return list(
        {
            token_id
            for token_id in range(tokenizer.vocab_size)
            if any(
                character in tokenizer.decode([token_id]).removeprefix(" ")
                for character in unwanted
            )
        }
    )


def clean_eninzh(text: str) -> str:
    return re.sub(r"(?<=[A-Za-z]) (?=[A-Za-z])", "", text)


def post_process(text: str, lang: str) -> str:
    from zhon.hanzi import punctuation

    for character in punctuation + string.punctuation:
        if character != "'":
            text = text.replace(character, "")
    text = re.sub(r" +", " ", text).strip()
    if lang == "zh":
        return clean_eninzh(" ".join(text))
    return text.lower()


def process_one(hypothesis: str, truth: str, lang: str):
    from jiwer import process_words

    truth = post_process(truth, lang)
    hypothesis = post_process(hypothesis, lang)
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


def _audio_name(reference: str, index=None) -> str:
    path = PurePosixPath(reference.replace("\\", "/"))
    name = f"{path.parent.name}_{path.stem}"
    return f"{name}_{index}" if index is not None else name


def _load_entries(args):
    entries = []
    with args.test_list.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.rstrip("\n").split("\t")
            if fields == [""]:
                continue
            if len(fields) == 4:
                candidates = [(_audio_name(fields[-2]), fields[-1])]
            elif len(fields) == 6:
                candidates = [
                    (_audio_name(fields[-4], index), text)
                    for index, text in enumerate(fields[-3:])
                    if text
                ]
            else:
                raise ValueError(
                    f"Invalid test-list line {line_number}: "
                    "expected 4 or 6 tab-separated fields"
                )
            for name, text in candidates:
                audio = args.wav_path / (
                    name + "." + args.extension.lstrip(".")
                )
                if not audio.is_file():
                    raise FileNotFoundError(f"Audio not found: {audio}")
                entries.append((audio, text))
    if not entries:
        raise ValueError("No evaluation utterances found")
    return entries


def main() -> int:
    args = get_parser().parse_args()

    import numpy as np
    import soundfile as sf
    import torch
    import zhconv
    from scipy.signal import resample
    from tqdm import tqdm

    if not args.test_list.is_file():
        raise FileNotFoundError(f"Test list not found: {args.test_list}")
    entries = _load_entries(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.lang == "en":
        from transformers import (
            GenerationConfig,
            WhisperForConditionalGeneration,
            WhisperProcessor,
        )

        model_path = args.model_dir / "wer" / "whisper-large-v3"
        if not model_path.is_dir():
            raise FileNotFoundError(f"Whisper model not found: {model_path}")
        processor = WhisperProcessor.from_pretrained(model_path)
        model = WhisperForConditionalGeneration.from_pretrained(
            model_path
        ).to(device)
        config = GenerationConfig.from_pretrained(model_path)
        suppressed = get_suppressed_tokens(processor.tokenizer)
        config.suppress_tokens = sorted(
            set((config.suppress_tokens or []) + suppressed)
        )
    else:
        from funasr import AutoModel

        model_path = args.model_dir / "wer" / "paraformer-zh"
        if not model_path.is_dir():
            raise FileNotFoundError(f"Paraformer model not found: {model_path}")
        model = AutoModel(model=str(model_path), disable_update=True)

    output = None
    if args.decode_path:
        args.decode_path.parent.mkdir(parents=True, exist_ok=True)
        output = args.decode_path.open("w", encoding="utf-8")

    substitutions = deletions = insertions = word_count = 0
    utterance_wers = []
    try:
        for audio_path, reference in tqdm(entries):
            if args.lang == "en":
                waveform, sample_rate = sf.read(audio_path)
                if sample_rate != 16000:
                    waveform = resample(
                        waveform, int(len(waveform) * 16000 / sample_rate)
                    )
                features = processor(
                    waveform,
                    sampling_rate=16000,
                    return_tensors="pt",
                ).input_features.to(device)
                predicted = model.generate(
                    features,
                    forced_decoder_ids=processor.get_decoder_prompt_ids(
                        language="english", task="transcribe"
                    ),
                    generation_config=config,
                )
                transcription = processor.batch_decode(
                    predicted, skip_special_tokens=True
                )[0]
            else:
                result = model.generate(
                    input=str(audio_path),
                    batch_size_s=300,
                    disable_pbar=True,
                )
                transcription = zhconv.convert(result[0]["text"], "zh-cn")

            truth, hypothesis, wer, subs, dels, ins, words = process_one(
                transcription, reference, args.lang
            )
            utterance_wers.append(float(wer))
            substitutions += subs
            deletions += dels
            insertions += ins
            word_count += words
            if output:
                output.write(
                    f"{audio_path}\t{wer}\t{truth}\t{hypothesis}\t"
                    f"{ins}\t{dels}\t{subs}\n"
                )
    finally:
        if output:
            output.close()

    seed_wer = round(float(np.mean(utterance_wers)) * 100, 2)
    corpus_wer = round(
        (substitutions + deletions + insertions) / word_count * 100, 2
    )
    logging.info("Seed-TTS WER: %.2f%%", seed_wer)
    logging.info("Corpus WER: %.2f%%", corpus_wer)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
