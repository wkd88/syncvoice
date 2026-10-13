# Third-party components

SyncVoice is based on or interoperates with the following projects. Upstream
licenses and redistribution terms apply to their code and weights even when
downloaded into `sup/` or `checkpoints/`.

## ZipVoice

- Source: https://github.com/k2-fsa/ZipVoice
- License: Apache License 2.0
- Role: flow-matching TTS backbone, tokenizer, feature extraction, and vocoder
  integration.

Modified ZipVoice files must retain their original copyright and license
headers and clearly identify local changes.

## Lipreading using Temporal Convolutional Networks

- Source:
  https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks
- Model zoo:
  https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks#model-zoo
- Role: mouth preprocessing and 512-D lip features.
- Restriction: the upstream project states that its supplied code is for
  comparative or benchmarking use and non-commercial purposes.

Its source is not included in the intended release and is cloned directly from
upstream by `tools/setup_visual_backends.py`. Do not assume that its source
code or pretrained weights can be relicensed under Apache-2.0.

## ibug face detection and alignment

- Sources: https://github.com/hhj1897/face_detection and
  https://github.com/hhj1897/face_alignment
- License: MIT
- Role: RetinaFace face detection and FAN 68-point landmark alignment.
- Weights typically used: `Resnet50_Final.pth`, `2dfan2.pth`.

Their source and weights are obtained from upstream by
`tools/setup_visual_backends.py`.

## EmoNet

- Source: https://github.com/face-analysis/emonet
- Role: 256-D facial-expression features.
- License stated upstream: CC BY-NC-ND 4.0.
- Weight typically used: `emonet_8.pth`.

CC BY-NC-ND restricts commercial use and distribution of adaptations. EmoNet
source is therefore not included in the intended release; users obtain it
directly from upstream with `tools/setup_visual_backends.py`. Confirm that
redistribution of its pretrained weights is permitted before hosting them.

## Vocos

- Source: https://github.com/charactr-platform/vocos
- Model: https://huggingface.co/charactr/vocos-mel-24khz
- License: MIT
- Role: converts generated acoustic features to 24 kHz waveforms.

Vocos is downloaded at runtime and is not distributed with this repository.

## Sync evaluation (LatentSync)

Audio-visual synchronization evaluation delegates to ByteDance's official
LatentSync implementation and does not vendor the legacy ZipVoice SyncNet tree.

- Source: https://github.com/bytedance/LatentSync
- License: Apache License 2.0
- Model repository: https://huggingface.co/ByteDance/LatentSync-1.6
- Checkpoints used for evaluation:
  - `auxiliary/syncnet_v2.model`
  - `auxiliary/sfd_face.pth`

LatentSync itself acknowledges prior SyncNet / face-detection work such as
[syncnet_python](https://github.com/joonson/syncnet_python) and related S3FD
detectors. Those upstream terms continue to apply to any derived evaluation
assets.

Install with `tools/setup_evaluation.py`.

## TTS evaluation models

SIM-o, UTMOS, and WER follow the evaluation layout published with ZipVoice:

- Evaluation scripts origin: https://github.com/k2-fsa/ZipVoice
- Model pack: https://huggingface.co/k2-fsa/TTS_eval_models
- Local download path: `checkpoints/evaluation/tts`

Migrated evaluation source retains its Apache-2.0 copyright headers. Individual
model files may have additional upstream terms, including:

| Metric | Upstream references |
|--------|---------------------|
| SIM-o (WavLM + ECAPA-TDNN) | [s3prl](https://github.com/s3prl/s3prl), [ECAPA-TDNN](https://github.com/lawlict/ECAPA-TDNN) |
| UTMOS | [SpeechMOS / UTMOS](https://github.com/tarepan/SpeechMOS) |
| English WER (Seed-TTS style) | [Whisper](https://github.com/openai/whisper), [transformers](https://github.com/huggingface/transformers) |
| Chinese WER (Seed-TTS style) | [FunASR Paraformer](https://github.com/modelscope/FunASR) |
| LibriSpeech-style WER | HuBERT ASR checkpoint in `TTS_eval_models` |

Install with `python tools/setup_evaluation.py --with-tts-models`.

## Model weights

The SyncVoice checkpoint, RetinaFace, FAN, lipreading, EmoNet, Vocos, LatentSync,
and TTS evaluation weights may each have different terms. Hosting files in one
Hugging Face repository does not replace those terms. Include the applicable
license and source metadata in the model card and do not upload a third-party
weight unless redistribution is allowed.

Evaluation model downloads and cached assets must not be committed.

