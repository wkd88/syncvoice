# Usage guide

[Back to README](../README.md) · [Evaluation](../EVALUATION.md)

Start with the [quick start](../README.md#quick-start). Commands below run from
the repository root in the activated `syncvoice` environment.

## Manual installation

Use this instead of `conda env create -f environment.yml`:

```bash
conda create -n syncvoice python=3.10 ffmpeg -c conda-forge
conda activate syncvoice
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Download the acoustic model with `python tools/download_weights.py`.
Use `--revision COMMIT_OR_TAG` to select a model revision, or `--output-dir PATH`
to change the download directory. Point inference's `--model-dir` to the
`lrs_exp` subdirectory containing the three model files.

## Prepare inputs

| Input | Purpose |
| --- | --- |
| `--prompt-wav` | Speaker reference audio. |
| `--prompt-text` | Transcript of that speaker reference. |
| `--text` | Text to synthesize. |
| `--video` | Target talking-face MP4 for visual conditioning. |
| `--wav` | Duration reference audio; defaults to the target video's same-stem WAV. |

The speaker prompt and duration reference serve different purposes. The current
video inference path reads a WAV even when the MP4 already contains audio. If
needed, extract the video's audio first:

```bash
ffmpeg -i data/clip.mp4 -vn -ac 1 -ar 24000 data/clip.wav
```

This requires an audio stream in the source video. For a silent video, supply
a suitable duration reference explicitly with `--wav`.

## Visual features

Skip setup when both cached feature files already exist beside each video.
Otherwise, install Git LFS and run:

```bash
python tools/setup_visual_backends.py
```

The script fetches Lipreading TCN, EmoNet, and ibug face tools into `sup/`, and
collects these weights in `checkpoints/visual/`:

- `Resnet50_Final.pth`
- `2dfan2.pth`
- `emonet_8.pth`
- `lrw_resnet18_dctcn_video.pth`

Setup needs access to GitHub and the upstream weight hosts, including Google
Drive. These components retain their [upstream terms](../THIRD_PARTY.md).

Inference extracts missing features automatically. To prepare them in advance:

```bash
python tools/extract_visual_features.py data/clip.mp4 \
  --weights-dir checkpoints/visual
```

You can also pass a directory. Extraction searches recursively by default;
`--no-recursive` limits it to that directory. Outputs are `<stem>.lip.npy`
(512-D lip features) and `<stem>.face.npy` (256-D face features). Existing
features are reused; use `--overwrite` after changing a source video.

## Inference controls

Add these options to the [quick-start command](../README.md#3-run-the-included-example):

| Option | Default | Purpose |
| --- | --- | --- |
| `--lip-guidance-scale` | `0.5` | Lip conditioning strength. |
| `--face-guidance-scale` | `0.5` | Facial-expression conditioning strength. |
| `--guidance-scale` | `1.0` | Classifier-free guidance; the quick start uses `0.8`. |
| `--num-step` | `16` | Sampling steps. |
| `--no-use-prompt-feature` | Off | Disable prompt acoustic/visual prefixes. |
| `--seed` | `666` | Random seed. |

The released checkpoint uses `--model-name zipvoice`, `--tokenizer emilia`,
and `--checkpoint-name checkpoint-55000.pt`; these are already the defaults.
For the full option list, run `python tools/infer_syncvoice.py --help`.

## Batch inference

Create a headerless `test.tsv` with four fields separated by actual tab characters:

```text
prompt_audio<TAB>prompt_text<TAB>target_audio_or_video<TAB>target_text
```

Use a target `.mp4` for visual conditioning; it needs a same-stem `.wav` and
cached features or configured visual backends. A target WAV alone uses the
audio duration reference without video conditioning. Use forward slashes and
include a parent directory in target paths, for example `data/clip.mp4`.

```bash
python tools/infer_syncvoice.py \
  --model-dir checkpoints/syncvoice/lrs_exp \
  --test-list test.tsv \
  --res-dir results/testset \
  --visual-weights-dir checkpoints/visual \
  --guidance-scale 0.8
```

Do not combine `--test-list` with the single-example prompt/text arguments.
Outputs use `<target_parent>_<target_stem>.wav`, with an MP4 for video inputs.
For example, `data/clip.mp4` produces `data_clip.wav` and `data_clip.mp4`.
Keep these parent/stem pairs unique to avoid overwriting outputs.

Six-field rows can supply three target texts after the first three fields;
their outputs receive `_0`, `_1`, and `_2` suffixes.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Missing `checkpoint-55000.pt`, `model.json`, or `tokens.txt` | Rerun `tools/download_weights.py`; `--model-dir` must point to `lrs_exp`. |
| `Conditioning audio not found` | Add a same-stem WAV beside the target video or pass `--wav` for single-example inference. |
| Missing or incomplete visual weights | Install Git LFS and rerun `tools/setup_visual_backends.py`; follow its reported upstream download path if needed. |
| WAV exists but video merging fails | Check that FFmpeg is on `PATH` and the source MP4 is readable. |
| Download fails on the first inference run | The vocoder also needs network access; downloading the SyncVoice checkpoint alone does not cache it. |

For bug reports, include the command, full error, Python/PyTorch/CUDA versions,
and whether you used cached visual features.
