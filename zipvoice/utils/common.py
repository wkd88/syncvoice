"""Minimal utilities required by SyncVoice inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import torch


class AttributeDict(dict):
    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(f"No such attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        if key in self:
            del self[key]
            return
        raise AttributeError(f"No such attribute '{key}'")

    def __str__(self, indent: int = 2):
        values = {}
        for key, value in self.items():
            if isinstance(value, (Path, torch.device, torch.dtype)):
                value = str(value)
            values[key] = value
        return json.dumps(values, indent=indent, sort_keys=True)


def prepare_avg_tokens_durations(
    features_lens: torch.Tensor, tokens_lens: torch.Tensor
) -> List[List[int]]:
    durations = []
    for feature_len, token_len in zip(features_lens, tokens_lens):
        average = int(feature_len // token_len)
        durations.append([average] * int(token_len))
    return durations


def pad_labels(
    labels: List[List[int]], pad_id: int, device: torch.device
) -> torch.Tensor:
    labels = [token_ids + [pad_id] for token_ids in labels]
    length = max(len(token_ids) for token_ids in labels)
    labels = [
        token_ids + [pad_id] * (length - len(token_ids))
        for token_ids in labels
    ]
    return torch.tensor(labels, dtype=torch.int64, device=device)


def get_tokens_index(
    durations: List[List[int]], num_frames: int
) -> torch.Tensor:
    durations = [item + [num_frames - sum(item)] for item in durations]
    answer = torch.zeros(len(durations), num_frames, dtype=torch.int64)
    for batch_index, item in enumerate(durations):
        current_frame = 0
        for token_index, duration in enumerate(item):
            answer[
                batch_index, current_frame : current_frame + duration
            ] = token_index
            current_frame += duration
        if current_frame != num_frames:
            raise RuntimeError(
                f"Token durations total {current_frame}, expected {num_frames}."
            )
    return answer


def make_pad_mask(
    lengths: torch.Tensor, max_len: int = 0
) -> torch.Tensor:
    if lengths.ndim != 1:
        raise ValueError(f"Expected 1-D lengths, got {lengths.ndim}-D.")
    length = max(max_len, int(lengths.max()))
    sequence = torch.arange(length, device=lengths.device)
    return sequence.unsqueeze(0).expand(
        lengths.size(0), length
    ) >= lengths.unsqueeze(-1)
