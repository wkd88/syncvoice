# `sup/` — upstream backends (not shipped in Git)

This directory is created and filled by setup scripts. Upstream code is
cloned here at runtime and is **not** redistributed with SyncVoice.

## Create / populate

From the repository root:

```bash
# Visual backends (Lipreading TCN, EmoNet, ibug face detection/alignment)
# Also copies extractor weights into checkpoints/visual/
python tools/setup_visual_backends.py
```

Optional evaluation backend:

```bash
# LatentSync -> sup/LatentSync (+ SyncNet checkpoints)
python tools/setup_evaluation.py
```

After setup you should see roughly:

```text
sup/
  Lipreading_using_Temporal_Convolutional_Networks/
  emonet/
  LatentSync/          # only after setup_evaluation.py
```

Requires network access to GitHub (and Git LFS for some visual weights).
See the root [README](../README.md) and [THIRD_PARTY.md](../THIRD_PARTY.md).
