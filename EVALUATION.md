# Evaluation

Optional objective metrics for SyncVoice outputs: perceptual quality (UTMOS),
speaker similarity (SIM-o), intelligibility (WER), and audio-visual
synchronization (LatentSync v2).

Audio metrics follow the ZipVoice evaluation protocol. Sync confidence uses
the official ByteDance LatentSync v2 evaluator. Upstream links are listed in
the README [References](README.md#references).

## Setup

```bash
pip install -r requirements-eval.txt
python tools/setup_evaluation.py
```

This creates `sup/` if needed, clones LatentSync into `sup/LatentSync`, and
downloads:

- `auxiliary/syncnet_v2.model`
- `auxiliary/sfd_face.pth`

from Hugging Face (`ByteDance/LatentSync-1.6`). If GitHub is unavailable, place
an official LatentSync checkout at `sup/LatentSync` and rerun the command to
fetch checkpoints only.

For SIM-o, UTMOS, or WER, also download the ZipVoice TTS evaluation models
(several GB):

```bash
python tools/setup_evaluation.py --with-tts-models
```

Models land under `checkpoints/evaluation/tts`.

## Audio-visual synchronization

Requires LatentSync's `syncnet_v2.model`. The legacy ZipVoice SyncNet path is
not used.

```bash
python -m zipvoice.eval.sync.latentsync_v2 \
  --video results/result.mp4
```

Directory of videos:

```bash
python -m zipvoice.eval.sync.latentsync_v2 \
  --input-dir results
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

## Test-list format

```text
prompt_audio<TAB>prompt_text<TAB>target_audio_or_video<TAB>target_text
```

Generated files follow inference naming: `<target_parent>_<target_stem>.wav`.
Seed-TTS WER and SIM-o also accept six-field multilingual rows with three
target texts.

When publishing results, report model revisions, SyncNet checkpoint, language,
sample count, and preprocessing protocol.
