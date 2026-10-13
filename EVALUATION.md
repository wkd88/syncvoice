# Evaluation

[Back to README](README.md)

Optional objective metrics for SyncVoice outputs: perceptual quality (UTMOS),
speaker similarity (SIM-o), intelligibility (WER), and audio-visual
synchronization (LatentSync v2).

Audio metrics follow the ZipVoice evaluation protocol. Sync confidence uses
the official ByteDance LatentSync v2 evaluator. Upstream sources and terms are
listed in [THIRD_PARTY.md](THIRD_PARTY.md).

## Setup

First complete the [installation](README.md#1-install). Run commands from the
repository root in the activated environment. Evaluation dependencies are not
needed for inference.

```bash
pip install -r requirements-eval.txt
python tools/setup_evaluation.py
```

This downloads the LatentSync backend and its evaluation checkpoints.

For SIM-o, UTMOS, or WER, also download the ZipVoice TTS evaluation models
(several GB):

```bash
python tools/setup_evaluation.py --with-tts-models
```

Models land under `checkpoints/evaluation/tts`.

## Audio-visual synchronization

```bash
python -m zipvoice.eval.sync.latentsync_v2 \
  --video results/result.mp4
```

- **Sync confidence**: higher means stronger audio–lip correspondence.
- **AV offset**: estimated offset in video frames; near zero is best.

LatentSync's v2 SyncNet was trained on its own aligned/preprocessed videos.
Compare systems only under the same preprocessing and evaluator version.

## UTMOS

Perceptual speech quality; higher is better.

```bash
python -m zipvoice.eval.mos.utmos \
  --wav-path results/testset \
  --model-dir checkpoints/evaluation/tts
```

## Speaker similarity

SIM-o compares each generated utterance with its prompt (WavLM + ECAPA-TDNN).
Higher cosine similarity is better.

```bash
python -m zipvoice.eval.speaker_similarity.sim \
  --wav-path results/testset \
  --test-list test.tsv \
  --model-dir checkpoints/evaluation/tts
```

## Word error rate

Seed-TTS-style English / Chinese:

```bash
python -m zipvoice.eval.wer.seedtts \
  --wav-path results/testset \
  --test-list test.tsv \
  --model-dir checkpoints/evaluation/tts \
  --lang en
```

Use `--lang zh` for Chinese. Lower WER is better.
English ASR: Whisper-large-v3. Chinese ASR: FunASR Paraformer.

LibriSpeech-style English:

```bash
python -m zipvoice.eval.wer.hubert \
  --wav-path results/testset \
  --test-list transcript.tsv \
  --model-dir checkpoints/evaluation/tts
```

## Test-list formats

SIM-o and Seed-TTS WER use a headerless TSV with actual tab separators:

```text
prompt_audio<TAB>prompt_text<TAB>target_audio_or_video<TAB>target_text
```

Place generated audio in `--wav-path`, named `<target_parent>_<target_stem>.wav`
to match the evaluator. For example, `data/clip.mp4` maps to `data_clip.wav`.

HuBERT WER uses a different, headerless `transcript.tsv`:

```text
generated_filename_without_extension<TAB>reference_text
```

For example, `data_clip<TAB>Reference words here` refers to
`results/testset/data_clip.wav`. UTMOS does not need a transcript list.


