# SyncVoice

**Simple and Effective Automatic Video Dubbing with Vision-Augmented TTS.**

[SyncVoice](https://arxiv.org/abs/2512.05126) builds on
[ZipVoice](https://github.com/k2-fsa/ZipVoice) to generate
speech from text, a speaker prompt, and a talking-face video. This release
provides an LRS3-trained checkpoint, inference tools, visual feature extraction,
and optional evaluation.

[Paper](https://arxiv.org/abs/2512.05126) ·
[Model weights](https://huggingface.co/kaidi555/SyncVoice) ·
[Usage guide](docs/USAGE.md) · [Evaluation](EVALUATION.md) ·
[Third-party components](THIRD_PARTY.md)

## Quick start

Use Linux with Python 3.10 and FFmpeg. An NVIDIA GPU with CUDA 12.1 is recommended;
the environment below installs PyTorch 2.5.1 and its CUDA 12.1 runtime.
Run all commands from the repository root.

### 1. Install

```bash
git clone https://github.com/wkd88/syncvoice.git
cd syncvoice
conda env create -f environment.yml
conda activate syncvoice
```

For an existing environment, see [manual installation](docs/USAGE.md#manual-installation).

### 2. Download the model

```bash
python tools/download_weights.py
```

This downloads `checkpoint-55000.pt`, `model.json`, and `tokens.txt` into
`checkpoints/syncvoice/lrs_exp/`. The vocoder is downloaded on first use, so keep
network access available for the first run.

### 3. Run the included example

The repository includes a video, its reference audio, and precomputed lip/face
features in `examples/`. No visual backend setup is needed for this example.

```bash
python tools/infer_syncvoice.py \
  --model-dir checkpoints/syncvoice/lrs_exp \
  --prompt-wav examples/target.wav \
  --prompt-text "You know, it is distinctive in many ways." \
  --text "You know, it is distinctive in many ways." \
  --video examples/target.mp4 \
  --res-wav-path results/result.wav \
  --guidance-scale 0.8
```

Outputs: `results/result.wav` and `results/result.mp4` (the input video with
generated audio). This smoke test reuses the target audio as the speaker prompt;
use a separate prompt to try another voice.

## Use your own video

1. Prepare a speaker WAV and its transcript, the target text, and a target MP4.
2. Place a same-stem WAV beside the target video (for example, `clip.mp4` and
   `clip.wav`), or pass `--wav path/to/reference.wav`. **The current video inference
   path requires this audio reference to determine the output duration.**
3. If the video has no cached `.lip.npy` / `.face.npy` files, install
   [Git LFS](https://git-lfs.github.com/) and run:

   ```bash
   python tools/setup_visual_backends.py
   ```

4. Adapt the quick-start command to your files and add
   `--visual-weights-dir checkpoints/visual`. Missing visual features are
   extracted automatically and saved beside the video.

See the [usage guide](docs/USAGE.md) for input preparation, batch inference,
guidance controls, and troubleshooting.

## Evaluation

The optional [evaluation guide](EVALUATION.md) covers audio-visual synchronization,
speech quality (UTMOS), speaker similarity (SIM-o), and word error rate (WER),
including model setup and input formats.

## License and acknowledgements

Original SyncVoice code is released under [Apache-2.0](LICENSE). The project
builds on ZipVoice and uses third-party visual and evaluation components.
Their code and weights retain their own terms, including non-commercial
restrictions for some visual backends. See [THIRD_PARTY.md](THIRD_PARTY.md)
and [NOTICE](NOTICE) for sources and attribution.
