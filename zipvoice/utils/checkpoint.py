"""Inference-only checkpoint loading."""

from __future__ import annotations

import logging
import os
import pathlib
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn


def load_checkpoint(
    filename: Path,
    model: Optional[nn.Module] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    logging.info("Loading checkpoint from %s", filename)
    original_posix_path = pathlib.PosixPath
    if os.name == "nt":
        pathlib.PosixPath = pathlib.WindowsPath
    try:
        checkpoint = torch.load(
            filename, map_location="cpu", weights_only=False
        )
    finally:
        pathlib.PosixPath = original_posix_path
    if model is None:
        return checkpoint

    state = checkpoint["model"]
    if state and next(iter(state)).startswith("module."):
        state = {
            key.removeprefix("module."): value for key, value in state.items()
        }
    incompatible = model.load_state_dict(state, strict=strict)
    if incompatible.missing_keys:
        logging.warning(
            "Checkpoint is missing %d model keys: %s",
            len(incompatible.missing_keys),
            ", ".join(incompatible.missing_keys),
        )
    if incompatible.unexpected_keys:
        logging.warning(
            "Checkpoint has %d unexpected keys: %s",
            len(incompatible.unexpected_keys),
            ", ".join(incompatible.unexpected_keys),
        )

    missing_visual = [
        key
        for key in incompatible.missing_keys
        if key.startswith(("tv_fusion.", "face_", "lip_"))
    ]
    if missing_visual:
        raise RuntimeError(
            "The checkpoint is missing required visual-conditioning weights: "
            + ", ".join(missing_visual)
        )
    return {
        key: value for key, value in checkpoint.items() if key != "model"
    }
