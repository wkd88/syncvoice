# SyncVoice

SyncVoice is a face- and lip-conditioned text-to-speech model built on
[ZipVoice](https://github.com/k2-fsa/ZipVoice). Given text, a speaker prompt,
and a talking-face video, it generates speech that better matches lip motion
and facial expression. The released checkpoint is trained on LRS3.

This repository covers inference, visual feature extraction, and optional
evaluation. Weights and test media are not stored in Git.

Weights: [kaidi555/SyncVoice](https://huggingface.co/kaidi555/SyncVoice)

## Requirements

- Linux
- Python 3.10
- NVIDIA GPU with CUDA 12.1 recommended
- FFmpeg on `PATH`
- [Git LFS](https://git-lfs.github.com/) (for visual extractor weights)

## Installation

```bash
conda env create -f environment.yml
conda activate syncvoice
```

Or:

```bash
conda create -n syncvoice python=3.10 ffmpeg -c conda-forge
conda activate syncvoice
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Run commands from the repository root.

```bash
# Creates sup/ and downloads visual backends + extractor weights
python tools/setup_visual_backends.py

# SyncVoice acoustic model -> checkpoints/syncvoice/lrs_exp/
python tools/download_weights.py
```

`sup/` is empty in Git on purpose. `setup_visual_backends.py` creates it and
clones Lipreading TCN, EmoNet, and ibug face tools there (see
[sup/README.md](sup/README.md)). Those trees stay under upstream licenses.
Visual weights are only needed when `.lip.npy` / `.face.npy` are missing.

## Extract visual features

```bash
python tools/extract_visual_features.py examples/target.mp4 \
  --weights-dir checkpoints/visual
```

Or pass a directory of MP4 files. Writes `*.lip.npy` / `*.face.npy` beside
each video (reused unless `--overwrite`).

## Inference

A demo clip is shipped under `examples/` (`target.mp4`, `target.wav`,
`target.lip.npy`, `target.face.npy`). Use your own `prompt.wav`, or reuse
`examples/target.wav` for a smoke test:

```bash
python tools/infer_syncvoice.py \
  --model-name zipvoice \
  --model-dir checkpoints/syncvoice/lrs_exp \
  --checkpoint-name checkpoint-55000.pt \
  --visual-weights-dir checkpoints/visual \
  --tokenizer emilia \
  --prompt-wav examples/target.wav \
  --prompt-text "You know, it is distinctive in many ways." \
  --text "You know, it is distinctive in many ways." \
  --video examples/target.mp4 \
  --res-wav-path results/result.wav \
  --face-guidance-scale 0.5 \
  --lip-guidance-scale 0.5 \
  --guidance-scale 0.8 \
  --num-step 16
```

Missing visual features are extracted automatically; audio is also muxed to
`results/result.mp4`.

- `--lip-guidance-scale` / `--face-guidance-scale`: visual conditioning strength
- `--guidance-scale`: text guidance strength
- `--no-use-prompt-feature`: disable prompt acoustic/visual prefixes (on by default)
- `--wav`: duration reference when no same-stem WAV sits beside the video

## Evaluation

```bash
pip install -r requirements-eval.txt
python tools/setup_evaluation.py
```

Details: [EVALUATION.md](EVALUATION.md).

## Project layout

```text
tools/                 # download, extract, infer, setup helpers
zipvoice/
  bin/  eval/  models/  visual/
sup/                   # created by setup scripts; see sup/README.md
checkpoints/           # downloaded weights (untracked)
EVALUATION.md  NOTICE  THIRD_PARTY.md
```

## License

Original SyncVoice code is Apache-2.0. Visual backends such as Lipreading TCN
and EmoNet have separate non-commercial or redistribution restrictions; see
[THIRD_PARTY.md](THIRD_PARTY.md) and [NOTICE](NOTICE).

## References

- ZipVoice: https://github.com/k2-fsa/ZipVoice
- SyncVoice weights: https://huggingface.co/kaidi555/SyncVoice
- Lipreading TCN: https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks
- EmoNet: https://github.com/face-analysis/emonet
- ibug face detection / alignment: https://github.com/hhj1897/face_detection , https://github.com/hhj1897/face_alignment
- LatentSync: https://github.com/bytedance/LatentSync
- LatentSync checkpoints: https://huggingface.co/ByteDance/LatentSync-1.6
- ZipVoice TTS evaluation models: https://huggingface.co/k2-fsa/TTS_eval_models
- SpeechMOS (UTMOS): https://github.com/tarepan/SpeechMOS
