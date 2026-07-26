"""Unified, non-destructive face and lip feature extraction.

Uses upstream backends cloned by ``tools/setup_visual_backends.py``:

- Lipreading TCN:
  https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks
- EmoNet: https://github.com/face-analysis/emonet
- ibug face detection: https://github.com/hhj1897/face_detection
- ibug face alignment: https://github.com/hhj1897/face_alignment
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import cv2
import numpy as np
import torch
import torchvision
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIP_ROOT = PROJECT_ROOT / "sup" / "Lipreading_using_Temporal_Convolutional_Networks"
EMONET_ROOT = PROJECT_ROOT / "sup" / "emonet"


def _add_vendor_paths() -> None:
    paths = (
        LIP_ROOT,
        LIP_ROOT / "face_detection",
        LIP_ROOT / "face_alignment",
        EMONET_ROOT,
    )
    for path in reversed(paths):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def _require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} not found: {path}\n"
            "Download the visual weights as described in README.md."
        )
    return path


def _nearest_fill(items: list[np.ndarray | None]) -> list[np.ndarray]:
    valid = [index for index, item in enumerate(items) if item is not None]
    if not valid:
        raise RuntimeError("No face was detected in the video.")
    result: list[np.ndarray] = []
    for index, item in enumerate(items):
        if item is not None:
            result.append(item)
            continue
        nearest = min(valid, key=lambda valid_index: abs(valid_index - index))
        result.append(items[nearest].copy())  # type: ignore[union-attr]
    return result


def _mouth_crops(
    frames: np.ndarray, landmarks: list[np.ndarray]
) -> np.ndarray:
    reference_path = LIP_ROOT / "preprocessing" / "20words_mean_face.npy"
    reference = np.load(_require_file(reference_path, "Mean-face template"))
    stable_points = (28, 33, 36, 39, 42, 45, 48, 54)
    crops = []

    for index, (frame, points) in enumerate(zip(frames, landmarks)):
        margin = min(6, index, len(landmarks) - 1 - index)
        smoothed = np.mean(
            landmarks[index - margin : index + margin + 1], axis=0
        )
        smoothed += points.mean(axis=0) - smoothed.mean(axis=0)
        transform = cv2.estimateAffinePartial2D(
            smoothed[list(stable_points)],
            reference[list(stable_points)],
            method=cv2.LMEDS,
        )[0]
        if transform is None:
            raise RuntimeError(f"Face alignment failed at frame {index}.")

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        aligned = cv2.warpAffine(gray, transform, (256, 256))
        aligned_points = (
            smoothed @ transform[:, :2].T + transform[:, 2]
        )
        mouth = aligned_points[48:68]
        center_x, center_y = np.mean(mouth, axis=0)
        x1, y1 = int(round(center_x - 48)), int(round(center_y - 48))
        x2, y2 = x1 + 96, y1 + 96
        if x1 < 0 or y1 < 0 or x2 > 256 or y2 > 256:
            raise RuntimeError(f"Mouth crop is out of bounds at frame {index}.")
        crops.append(aligned[y1:y2, x1:x2])

    return np.stack(crops)


def _lip_model(config_path: Path, model_path: Path, device: torch.device):
    from lipreading.model import Lipreading
    from lipreading.utils import load_model

    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)

    tcn_options = {}
    if config.get("tcn_num_layers"):
        tcn_options = {
            "num_layers": config["tcn_num_layers"],
            "kernel_size": config["tcn_kernel_size"],
            "dropout": config["tcn_dropout"],
            "dwpw": config["tcn_dwpw"],
            "width_mult": config["tcn_width_mult"],
        }

    densetcn_options = {}
    if config.get("densetcn_block_config"):
        densetcn_options = {
            "block_config": config["densetcn_block_config"],
            "growth_rate_set": config["densetcn_growth_rate_set"],
            "reduced_size": config["densetcn_reduced_size"],
            "kernel_size_set": config["densetcn_kernel_size_set"],
            "dilation_size_set": config["densetcn_dilation_size_set"],
            "squeeze_excitation": config["densetcn_se"],
            "dropout": config["densetcn_dropout"],
        }

    model = Lipreading(
        modality="video",
        num_classes=500,
        tcn_options=tcn_options,
        densetcn_options=densetcn_options,
        backbone_type=config["backbone_type"],
        relu_type=config["relu_type"],
        width_mult=config["width_mult"],
        use_boundary=False,
        extract_feats=True,
    ).to(device)
    model = load_model(str(model_path), model)
    model.eval()
    return model


def _emonet_embedding(model: torch.nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    """Return the 256-D representation before EmoNet's prediction head."""
    value = F.relu(model.bn1(model.conv1(tensor)), True)
    value = F.max_pool2d(model.conv2(value), 2, stride=2)
    value = model.conv3(value)
    value = model.conv4(value)
    previous = value
    hourglass_features = []
    for index in range(model.num_modules):
        level = model._modules[f"m{index}"](previous)
        level = model._modules[f"top_m_{index}"](level)
        level = F.relu(
            model._modules[f"bn_end{index}"](
                model._modules[f"conv_last{index}"](level)
            ),
            True,
        )
        heatmap = model._modules[f"l{index}"](level)
        if index < model.num_modules - 1:
            previous = (
                previous
                + model._modules[f"bl{index}"](level)
                + model._modules[f"al{index}"](heatmap)
            )
        hourglass_features.append(level)
    combined = torch.cat(tuple(hourglass_features), dim=1)
    if model.attention:
        combined *= torch.sum(heatmap, dim=1, keepdim=True)
        emotion = torch.cat((value, combined), dim=1)
    else:
        emotion = torch.cat((value, combined, heatmap), dim=1)
    emotion = model.conv1x1_input_emo_2(emotion)
    emotion = model.emo_net_2(emotion)
    emotion = model.avg_pool_2(emotion)
    return emotion.flatten(1)


class VisualFeatureExtractor:
    """Extract 512-D lip and 256-D face features from one video."""

    def __init__(
        self,
        weights_dir: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        _add_vendor_paths()
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        weights = Path(weights_dir) if weights_dir else None

        def weight(name: str, fallback: Path) -> Path:
            candidate = weights / name if weights else fallback
            return _require_file(candidate, name)

        retina_path = weight(
            "Resnet50_Final.pth",
            LIP_ROOT
            / "face_detection/ibug/face_detection/retina_face/weights"
            / "Resnet50_Final.pth",
        )
        fan_path = weight(
            "2dfan2.pth",
            LIP_ROOT
            / "face_alignment/ibug/face_alignment/fan/weights"
            / "2dfan2.pth",
        )
        lip_path = weight(
            "lrw_resnet18_dctcn_video.pth",
            LIP_ROOT / "models/lrw_resnet18_dctcn_video.pth",
        )
        emonet_path = weight(
            "emonet_8.pth",
            EMONET_ROOT / "pretrained/emonet_8.pth",
        )
        lip_config = LIP_ROOT / "configs/lrw_resnet18_dctcn.json"

        from emonet.models import EmoNet
        from ibug.face_alignment import FANPredictor
        from ibug.face_detection import RetinaFacePredictor

        retina_model = RetinaFacePredictor.get_model("resnet50")
        retina_model = SimpleNamespace(
            weights=str(retina_path), config=retina_model.config
        )
        fan_model = FANPredictor.get_model("2dfan2")
        fan_model = SimpleNamespace(weights=str(fan_path), config=fan_model.config)

        self.face_detector = RetinaFacePredictor(
            device=self.device, threshold=0.8, model=retina_model
        )
        self.landmark_detector = FANPredictor(
            device=self.device, model=fan_model
        )
        self.lip_model = _lip_model(lip_config, lip_path, self.device)

        state = torch.load(emonet_path, map_location="cpu", weights_only=True)
        state = {key.replace("module.", ""): value for key, value in state.items()}
        self.face_model = EmoNet(n_expression=8).to(self.device)
        self.face_model.load_state_dict(state, strict=False)
        self.face_model.eval()

    @torch.inference_mode()
    def _detect(
        self, frames: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        boxes: list[np.ndarray | None] = []
        landmarks: list[np.ndarray | None] = []
        for frame in frames:
            detections = self.face_detector(frame, rgb=True)
            if len(detections) == 0:
                boxes.append(None)
                landmarks.append(None)
                continue
            sizes = (detections[:, 2] - detections[:, 0]) * (
                detections[:, 3] - detections[:, 1]
            )
            index = int(np.argmax(sizes))
            points, _ = self.landmark_detector(frame, detections, rgb=True)
            boxes.append(detections[index, :4].copy())
            landmarks.append(points[index].copy())
        return _nearest_fill(boxes), _nearest_fill(landmarks)

    @torch.inference_mode()
    def extract(self, video_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")
        frames = torchvision.io.read_video(
            str(video_path), pts_unit="sec", output_format="THWC"
        )[0].numpy()
        if len(frames) == 0:
            raise RuntimeError(f"No frames decoded from: {video_path}")

        boxes, landmarks = self._detect(frames)
        mouth_video = _mouth_crops(frames, landmarks)
        lip_input = (
            torch.from_numpy(mouth_video)
            .float()
            .to(self.device)
            .unsqueeze(0)
            .unsqueeze(0)
            / 255.0
        )
        lip_input = lip_input[:, :, :, 4:92, 4:92]
        lip_input = (lip_input - 0.421) / 0.165
        lip = self.lip_model(
            lip_input, lengths=[lip_input.shape[2]]
        ).squeeze(0)

        face_features = []
        for frame, box in zip(frames, boxes):
            height, width = frame.shape[:2]
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, width), min(y2, height)
            if x2 <= x1 or y2 <= y1:
                raise RuntimeError("Detected an invalid face bounding box.")
            crop = cv2.resize(frame[y1:y2, x1:x2], (256, 256))
            tensor = (
                torch.from_numpy(crop)
                .permute(2, 0, 1)
                .float()
                .unsqueeze(0)
                .to(self.device)
                / 255.0
            )
            face_features.append(
                _emonet_embedding(self.face_model, tensor).squeeze(0)
            )

        face = torch.stack(face_features)
        return (
            lip.detach().cpu().float().numpy(),
            face.detach().cpu().float().numpy(),
        )

    def extract_to_files(
        self, video_path: str | Path, overwrite: bool = False
    ) -> tuple[Path, Path]:
        video_path = Path(video_path)
        base = video_path.with_suffix("")
        lip_path = Path(f"{base}.lip.npy")
        face_path = Path(f"{base}.face.npy")
        if not overwrite and lip_path.is_file() and face_path.is_file():
            return lip_path, face_path
        lip, face = self.extract(video_path)
        np.save(lip_path, lip)
        np.save(face_path, face)
        return lip_path, face_path


def extract_visual_features(
    video_path: str | Path,
    weights_dir: str | Path | None = None,
    device: str | torch.device | None = None,
    save: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    extractor = VisualFeatureExtractor(weights_dir=weights_dir, device=device)
    lip, face = extractor.extract(video_path)
    if save:
        base = Path(video_path).with_suffix("")
        np.save(Path(f"{base}.lip.npy"), lip)
        np.save(Path(f"{base}.face.npy"), face)
    return lip, face


def iter_videos(path: str | Path, recursive: bool = True) -> Iterable[Path]:
    path = Path(path)
    if path.is_file():
        yield path
        return
    pattern = "**/*.mp4" if recursive else "*.mp4"
    yield from sorted(path.glob(pattern))
