# Usage guide

[Back to README](../README.md) · [Evaluation](../EVALUATION.md)

Complete the [quick start](../README.md#quick-start) first. Run commands from
the repository root in the activated `syncvoice` environment.

## Prepare inputs

Use a speaker WAV and its transcript as `--prompt-wav` and `--prompt-text`.
Set `--text` to the text to synthesize and `--video` to the target MP4.

Video inference also requires a duration-reference WAV: place `clip.wav` beside
`clip.mp4`, or pass its path with `--wav`. If the video contains audio, extract it:

```bash
ffmpeg -i data/clip.mp4 -vn -ac 1 -ar 24000 data/clip.wav
```

## Visual features

If the video has no cached `.lip.npy` and `.face.npy` files, install
[Git LFS](https://git-lfs.github.com/) and set up the visual backends:

```bash
python tools/setup_visual_backends.py
```

Add `--visual-weights-dir checkpoints/visual` to your inference command.
Missing features are extracted automatically and saved beside the video.
To extract them separately:

```bash
python tools/extract_visual_features.py data/clip.mp4 \
  --weights-dir checkpoints/visual
```

Use `--overwrite` when re-extracting features after changing the source video.

## Adjust generation

- `--lip-guidance-scale` and `--face-guidance-scale`: visual conditioning strength
  (both default to `0.5`).
- `--guidance-scale`: classifier-free guidance (`0.8` in the quick-start example).
- `--num-step`: sampling steps (default: `16`).

For all options, run `python tools/infer_syncvoice.py --help`.

