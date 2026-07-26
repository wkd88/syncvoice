#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.
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

import logging
from typing import Optional, Union

import librosa
import numpy as np
import soundfile as sf
import torch


def load_waveform(
    fname: str,
    sample_rate: int,
    dtype: str = "float32",
    device: torch.device = torch.device("cpu"),
    return_numpy: bool = False,
    max_seconds: Optional[float] = None,
) -> Union[torch.Tensor, np.ndarray]:
    """Load an audio file, mix it to mono, and resample it if needed."""
    wav_data, sr = sf.read(fname, dtype=dtype)
    if wav_data.ndim == 2:
        wav_data = wav_data.mean(axis=1)
    if sr != sample_rate:
        wav_data = librosa.resample(
            wav_data, orig_sr=sr, target_sr=sample_rate
        )
    if max_seconds is not None:
        max_length = int(sample_rate * max_seconds)
        if len(wav_data) > max_length:
            wav_data = wav_data[:max_length]
            logging.warning(
                "Audio %s was truncated to %.1f seconds to avoid OOM.",
                fname,
                max_seconds,
            )
    if return_numpy:
        return wav_data
    return torch.from_numpy(wav_data).to(device)
