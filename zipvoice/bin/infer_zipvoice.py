#!/usr/bin/env python3
# Copyright         2025  Xiaomi Corp.        (authors: Han Zhu)
# Copyright         2026  SyncVoice contributors
#
# See NOTICE and THIRD_PARTY.md for third-party attribution.
# See LICENSE for the Apache License, Version 2.0 terms.
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

"""
Face- and lip-conditioned SyncVoice inference.

Download the SyncVoice acoustic model first::

    python tools/download_weights.py

Then pass a local ``--model-dir``. This entry point does not silently download
ZipVoice checkpoints.

Each line of ``test.tsv`` is in the format of
``prompt_wav\\tprompt_text\\ttarget_audio_or_video\\ttarget_text``.
"""

import argparse
import datetime as dt
import json
import logging
import os
from pathlib import Path
from typing import Optional

import subprocess as sp
import numpy as np
import safetensors.torch
import torch
import torchaudio
import torch.nn.functional as F
from lhotse.utils import fix_random_seed
from vocos import Vocos

from zipvoice.models.zipvoice import ZipVoice
from zipvoice.tokenizer.tokenizer import (
    EmiliaTokenizer,
    EspeakTokenizer,
    LibriTTSTokenizer,
    SimpleTokenizer,
)
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.common import AttributeDict
from zipvoice.utils.feature import VocosFbank
from zipvoice.visual import extract_visual_features


def merge_video_audio(video_path, audio_path, output_path):
    command = [
        'ffmpeg',
        '-hide_banner',
        '-loglevel', 'error',
        '-y',
        '-i', video_path,
        '-i', audio_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-shortest',
        '-strict', 'experimental',
        output_path
    ]
    
    try:
        sp.run(command, check=True)
        print(f"Successfully merged audio and video into {output_path}")
        return output_path
    except sp.CalledProcessError as e:
        print(f"Error merging audio and video: {e}")
        return None

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="zipvoice",
        choices=["zipvoice"],
        help="The model used for inference",
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Local SyncVoice model directory containing the checkpoint, "
        "model.json, and tokens.txt. Download with tools/download_weights.py.",
    )

    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="checkpoint-55000.pt",
        help="The name of the SyncVoice model checkpoint.",
    )

    parser.add_argument(
        "--vocoder-path",
        type=str,
        default=None,
        help="The vocoder checkpoint. "
        "Will download pre-trained vocoder from huggingface if not specified.",
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default="emilia",
        choices=["emilia", "libritts", "espeak", "simple"],
        help="Tokenizer type.",
    )

    parser.add_argument(
        "--lang",
        type=str,
        default="en-us",
        help="Language identifier, used when tokenizer type is espeak. see"
        "https://github.com/rhasspy/espeak-ng/blob/master/docs/languages.md",
    )

    parser.add_argument(
        "--test-list",
        type=str,
        default=None,
        help="TSV list for batch inference. Each line is "
        "'prompt_wav\\tprompt_text\\ttarget_audio_or_video\\ttarget_text'.",
    )

    parser.add_argument(
        "--prompt-wav",
        type=str,
        default=None,
        help="The prompt wav to mimic",
    )

    parser.add_argument(
        "--prompt-text",
        type=str,
        default=None,
        help="The transcription of the prompt wav",
    )

    parser.add_argument(
        "--prompt-video",
        type=str,
        default=None,
        help="The path of the prompt video",
    )

    parser.add_argument(
        "--wav",
        type=str,
        default=None,
        help="The path of the wav",
    )

    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="The path of the video",
    )

    parser.add_argument(
        "--visual-weights-dir",
        type=str,
        default=None,
        help="Optional directory containing face and lip extractor weights.",
    )

    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="The text to synthesize",
    )

    parser.add_argument(
        "--res-dir",
        type=str,
        default="results",
        help="""
        Path name of the generated wavs dir,
        used when test-list is not None
        """,
    )

    parser.add_argument(
        "--res-wav-path",
        type=str,
        default="result.wav",
        help="""
        Path name of the generated wav path,
        used when test-list is None
        """,
    )

    parser.add_argument(
        "--use-global-speaker",
        action="store_true",
        default=False,
        help="Use global speaker for inference",
    )

    parser.add_argument(
        "--use-prompt-feature",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep prompt acoustic/visual prefixes during inference "
            "(enabled by default; disable with --no-use-prompt-feature)."
        ),
    )

    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="The scale of classifier-free guidance during inference.",
    )

    parser.add_argument(
        "--video-guidance-scale",
        type=float,
        default=-5.0,
        help="Legacy combined visual guidance; -5.0 disables this branch.",
    )

    parser.add_argument(
        "--lip-guidance-scale",
        type=float,
        default=0.5,
        help="Classifier-free guidance strength for lip conditioning.",
    )

    parser.add_argument(
        "--face-guidance-scale",
        type=float,
        default=0.5,
        help="Classifier-free guidance strength for face conditioning.",
    )

    parser.add_argument(
        "--num-step",
        type=int,
        default=None,
        help="The number of sampling steps.",
    )

    parser.add_argument(
        "--feat-scale",
        type=float,
        default=0.1,
        help="The scale factor of fbank feature",
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Control speech speed, 1.0 means normal, >1.0 means speed up",
    )

    parser.add_argument(
        "--t-shift",
        type=float,
        default=0.5,
        help="Shift t to smaller ones if t_shift < 1.0",
    )

    parser.add_argument(
        "--target-rms",
        type=float,
        default=0.1,
        help="Target speech normalization rms value, set to 0 to disable normalization",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=666,
        help="Random seed",
    )

    return parser


def get_vocoder(vocos_local_path: Optional[str] = None):
    if vocos_local_path:
        vocoder = Vocos.from_hparams(f"{vocos_local_path}/config.yaml")
        state_dict = torch.load(
            f"{vocos_local_path}/pytorch_model.bin",
            weights_only=True,
            map_location="cpu",
        )
        vocoder.load_state_dict(state_dict)
    else:
        vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    return vocoder

def extract_reference_mel_from_speech(
    features,
    features_lens,
    max_length=6.0,
    sample_rate=24000,
    hop_size=256,
):
    """Center-crop prompt mels for inference-only speaker conditioning."""
    samples = []

    for feature, feature_len in zip(features, features_lens):
        sample = {
            "reference_mels": [],
            "reference_mel_lengths": [],
            "num_references": 0,
        }
        feature = feature[:feature_len, :]

        max_samples = int(max_length * sample_rate / hop_size)

        if feature_len > max_samples:
            start_idx = (feature_len - max_samples) // 2
            feature_segment = feature[start_idx : start_idx + max_samples, :]
        else:
            feature_segment = feature

        sample['reference_mels'] = [feature_segment]
        sample['reference_mel_lengths'] = [feature_segment.shape[0]]
        sample['num_references'] = 1
        
        samples.append(sample)
    
    # Pad reference mels into a rectangular batch.
    batch_size = len(samples)
    max_num_refs = max([sample['num_references'] for sample in samples], default=0)
    
    if max_num_refs == 0:
        # No valid references: return empty tensors with a consistent rank.
        padded_reference_mels = torch.empty(batch_size, 0, 0, 0, device=features[0].device)
        padded_reference_mel_lengths = torch.empty(batch_size, 0, dtype=torch.int32, device=features[0].device)
        reference_mel_masks = torch.empty(batch_size, 0, 0, device=features[0].device)
        return padded_reference_mels, padded_reference_mel_lengths, reference_mel_masks
    
    max_mel_length = 0
    mel_dim = features[0].shape[1]
    
    for sample in samples:
        for ref_mel in sample['reference_mels']:
            if ref_mel.numel() > 0:
                max_mel_length = max(max_mel_length, ref_mel.shape[0])
    
    # Shape: [B, N, T, D]
    padded_reference_mels = torch.zeros(
        batch_size, max_num_refs, max_mel_length, mel_dim,
        device=features[0].device, dtype=features[0].dtype
    )
    
    # Shape: [B, N]
    padded_reference_mel_lengths = torch.zeros(
        batch_size, max_num_refs, dtype=torch.int32, device=features[0].device
    )
    
    # Shape: [B, N, T]
    reference_mel_masks = torch.zeros(
        batch_size, max_num_refs, max_mel_length,
        device=features[0].device
    )
    
    for b_idx, sample in enumerate(samples):
        ref_mels = sample['reference_mels']
        ref_lengths = sample['reference_mel_lengths']
        
        for r_idx in range(min(len(ref_mels), max_num_refs)):
            if r_idx < len(ref_mels) and ref_mels[r_idx].numel() > 0:
                mel = ref_mels[r_idx]
                length = ref_lengths[r_idx] if r_idx < len(ref_lengths) else mel.shape[0]
                actual_length = min(length, mel.shape[0], max_mel_length)
                
                # Keep mel layout as [T, D].
                padded_reference_mels[b_idx, r_idx, :actual_length, :] = mel[:actual_length, :]
                padded_reference_mel_lengths[b_idx, r_idx] = actual_length
                reference_mel_masks[b_idx, r_idx, :actual_length] = 1.0
    
    return padded_reference_mels, padded_reference_mel_lengths, reference_mel_masks

def generate_sentence(
    save_path: str,
    prompt_text: str,
    prompt_wav: str,
    prompt_video: str,
    text: str,
    wav: str,
    video: str,
    model: torch.nn.Module,
    vocoder: torch.nn.Module,
    tokenizer: EmiliaTokenizer,
    feature_extractor: VocosFbank,
    device: torch.device,
    num_step: int = 16,
    guidance_scale: float = 1.0,
    video_guidance_scale: float = 0.0,
    face_guidance_scale: float = 0.0,
    lip_guidance_scale: float = 0.0,
    speed: float = 1.0,
    t_shift: float = 0.5,
    target_rms: float = 0.1,
    feat_scale: float = 0.1,
    sampling_rate: int = 24000,
    params: AttributeDict = AttributeDict(),
):
    """
    Generate waveform of a text based on a given prompt
        waveform and its transcription.

    Args:
        save_path (str): Path to save the generated wav.
        prompt_text (str): Transcription of the prompt wav.
        prompt_wav (str): Path to the prompt wav file.
        text (str): Text to be synthesized into a waveform.
        model (torch.nn.Module): The model used for generation.
        vocoder (torch.nn.Module): The vocoder used to convert features to waveforms.
        tokenizer (EmiliaTokenizer): The tokenizer used to convert text to tokens.
        feature_extractor (VocosFbank): The feature extractor used to
            extract acoustic features.
        device (torch.device): The device on which computations are performed.
        num_step (int, optional): Number of steps for decoding. Defaults to 16.
        guidance_scale (float, optional): Scale for classifier-free guidance.
            Defaults to 1.0.
        speed (float, optional): Speed control. Defaults to 1.0.
        t_shift (float, optional): Time shift. Defaults to 0.5.
        target_rms (float, optional): Target RMS for waveform normalization.
            Defaults to 0.1.
        feat_scale (float, optional): Scale for features.
            Defaults to 0.1.
        sampling_rate (int, optional): Sampling rate for the waveform.
            Defaults to 24000.
    Returns:
        metrics (dict): Dictionary containing time and real-time
            factor metrics for processing.
    """
    # Convert text to tokens
    tokens = tokenizer.texts_to_token_ids([text])
    prompt_tokens = tokenizer.texts_to_token_ids([prompt_text])

    # Load and preprocess prompt wav
    prompt_wav_path = prompt_wav
    prompt_wav, prompt_sampling_rate = torchaudio.load(prompt_wav)

    if prompt_sampling_rate != sampling_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=prompt_sampling_rate, new_freq=sampling_rate
        )
        prompt_wav = resampler(prompt_wav)

    prompt_rms = torch.sqrt(torch.mean(torch.square(prompt_wav)))
    if prompt_rms < target_rms:
        prompt_wav = prompt_wav * target_rms / prompt_rms

    # Extract features from prompt wav
    prompt_features = feature_extractor.extract(
        prompt_wav, sampling_rate=sampling_rate
    ).to(device)

    prompt_features = prompt_features.unsqueeze(0) * feat_scale
    prompt_features_lens = torch.tensor([prompt_features.size(1)], device=device)

    features_lens = None
    if wav is not None:
        g_wav, g_sampling_rate = torchaudio.load(wav)

        if g_sampling_rate != sampling_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=g_sampling_rate, new_freq=sampling_rate
            )
            g_wav = resampler(g_wav)
        g_features = feature_extractor.extract(
            g_wav, sampling_rate=sampling_rate
        ).to(device)

        g_features = g_features.unsqueeze(0) * feat_scale
        features_lens = torch.tensor([g_features.size(1)], device=device)

    lip_prompt, face_prompt = None, None
    if prompt_video is not None or video is not None:
        assert video is not None
        prompt_mel_len = prompt_features.size(1)
        lip_shape = (prompt_mel_len, 512)
        face_shape = (prompt_mel_len, 256) 
        p_lip_prompt, p_face_prompt = torch.zeros(lip_shape), torch.zeros(face_shape)

        base = os.path.splitext(video)[0]
        wav_path = wav or f"{base}.wav"
        if not os.path.isfile(wav_path):
            raise FileNotFoundError(
                f"Conditioning audio not found: {wav_path}. "
                "Pass --wav or place a same-stem WAV beside the video."
            )
        gt_wav, gt_sampling_rate = torchaudio.load(wav_path)
        if gt_sampling_rate != sampling_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=gt_sampling_rate, new_freq=sampling_rate
            )
            gt_wav = resampler(gt_wav)

        gt_rms = torch.sqrt(torch.mean(torch.square(gt_wav)))
        if gt_rms < target_rms:
            gt_wav = gt_wav * target_rms / gt_rms

        # Extract features from gt wav
        gt_features = feature_extractor.extract(
            gt_wav, sampling_rate=sampling_rate
        ).to(device)

        gt_features = gt_features.unsqueeze(0) * feat_scale
        features_lens = torch.tensor([gt_features.size(1)], device=device)
        gt_mel_len = gt_features.size(1)
    
        lipfeat_path = f"{base}.lip.npy"
        facefeat_path = f"{base}.face.npy"
        # imagefeat_path = f"{base}.image.npy"

        if not os.path.exists(lipfeat_path) or not os.path.exists(facefeat_path):
            lipfeat, facefeat = extract_visual_features(
                video,
                weights_dir=params.get("visual_weights_dir"),
                device=device,
                save=True,
            )
        else:
            lipfeat = np.load(lipfeat_path)
            facefeat = np.load(facefeat_path)
        lipfeat = torch.from_numpy(lipfeat)
        facefeat = torch.from_numpy(facefeat)
        lipfeat_final = F.interpolate(lipfeat.unsqueeze(0).transpose(1, 2),  # (1, 2*T, N)
                                size=gt_mel_len,
                                mode='nearest').transpose(1, 2).squeeze(0)  # (N, mellen)
        facefeat_final = F.interpolate(facefeat.unsqueeze(0).transpose(1, 2),
                                    size=gt_mel_len,
                                    mode='nearest').transpose(1, 2).squeeze(0)
        g_lip_prompt, g_face_prompt = lipfeat_final, facefeat_final # (T, N)

        if not params["use_prompt_feature"]:
            lip_prompt = g_lip_prompt.unsqueeze(0).to(device)
            face_prompt = g_face_prompt.unsqueeze(0).to(device)
        else:
            lip_prompt = torch.cat([p_lip_prompt, g_lip_prompt], dim=0).unsqueeze(0).to(device)
            face_prompt = torch.cat([p_face_prompt, g_face_prompt], dim=0).unsqueeze(0).to(device)

    spk_features = None
    reference_mels, reference_mel_masks = None, None
    if params.get("use_spk", False) and params["use_global_speaker"]:
        if params.get("use_pretrain_spk", False):
            spk_npy_features = []
            spk_path = prompt_wav_path.replace(".wav", ".spk.npy")
            spk_npy_feat = torch.from_numpy(np.load(spk_path)) # (feature_dim)
            spk_npy_features.append(spk_npy_feat) 
            spk_npy_features = torch.stack(spk_npy_features, dim=0) # (N, feature_dim)
            spk_npy_features = F.normalize(spk_npy_features, p=2, dim=1)
            spk_features = spk_npy_features.to(device)
        else:
            if params.get("use_spk_encoder_pretrain", False):
                spk_npy_features = []
                spk_path = prompt_wav_path.replace(".wav", ".spk.npy")
                spk_npy_feat = torch.from_numpy(np.load(spk_path)) # (feature_dim)
                spk_npy_features.append(spk_npy_feat) 
                spk_npy_features = torch.stack(spk_npy_features, dim=0) # (N, feature_dim)
                spk_npy_features = F.normalize(spk_npy_features, p=2, dim=1)
                spk_features = spk_npy_features.to(device)
            (
                reference_mels,
                reference_mel_lengths,
                reference_mel_masks,
            ) = extract_reference_mel_from_speech(
                prompt_features, prompt_features_lens
            )
    
    if not params["use_prompt_feature"]:
        prompt_tokens, prompt_features_lens, prompt_features = None, None, None

    # Start timing
    start_t = dt.datetime.now()

    if video is not None or wav is not None:
        duration = "real"
    else:
        duration = "predict"
    # Generate features
    (
        pred_features,
        pred_features_lens,
        pred_prompt_features,
        pred_prompt_features_lens,
    ) = model.sample(
        tokens=tokens,
        prompt_tokens=prompt_tokens,
        prompt_features=prompt_features,
        prompt_features_lens=prompt_features_lens,
        features_lens=features_lens,
        lip_features=lip_prompt,
        face_features=face_prompt,
        spk_features=spk_features,
        reference_mels=reference_mels,
        reference_mel_masks=reference_mel_masks,
        speed=speed,
        t_shift=t_shift,
        duration=duration, # predict real
        num_step=num_step,
        guidance_scale=guidance_scale,
        video_guidance_scale=video_guidance_scale,
        face_guidance_scale=face_guidance_scale,
        lip_guidance_scale=lip_guidance_scale,
    )

    # Postprocess predicted features
    pred_features = pred_features.permute(0, 2, 1) / feat_scale  # (B, C, T)

    # Start vocoder processing
    start_vocoder_t = dt.datetime.now()
    wav = vocoder.decode(pred_features).squeeze(1).clamp(-1, 1)

    # Calculate processing times and real-time factors
    t = (dt.datetime.now() - start_t).total_seconds()
    t_no_vocoder = (start_vocoder_t - start_t).total_seconds()
    t_vocoder = (dt.datetime.now() - start_vocoder_t).total_seconds()
    wav_seconds = wav.shape[-1] / sampling_rate
    rtf = t / wav_seconds
    rtf_no_vocoder = t_no_vocoder / wav_seconds
    rtf_vocoder = t_vocoder / wav_seconds
    metrics = {
        "t": t,
        "t_no_vocoder": t_no_vocoder,
        "t_vocoder": t_vocoder,
        "wav_seconds": wav_seconds,
        "rtf": rtf,
        "rtf_no_vocoder": rtf_no_vocoder,
        "rtf_vocoder": rtf_vocoder,
    }

    # Adjust wav volume if necessary
    if prompt_rms < target_rms:
        wav = wav * prompt_rms / target_rms
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(save_path, wav.cpu(), sample_rate=sampling_rate)

    if video is not None:
        base = os.path.splitext(save_path)[0]
        output_path = f"{base}.mp4"
        if os.path.exists(video):
            merge_video_audio(video, save_path, output_path)
    return metrics


def generate_list(
    res_dir: str,
    test_list: str,
    model: torch.nn.Module,
    vocoder: torch.nn.Module,
    tokenizer: EmiliaTokenizer,
    feature_extractor: VocosFbank,
    device: torch.device,
    num_step: int = 16,
    video_guidance_scale: float = 1.0,
    face_guidance_scale: float = 1.0,
    lip_guidance_scale: float = 1.0,
    guidance_scale: float = 1.0,
    speed: float = 1.0,
    t_shift: float = 0.5,
    target_rms: float = 0.1,
    feat_scale: float = 0.1,
    sampling_rate: int = 24000,
    params: AttributeDict = AttributeDict(),
):
    total_t = []
    total_t_no_vocoder = []
    total_t_vocoder = []
    total_wav_seconds = []

    with open(test_list, "r") as fr:
        lines = fr.readlines()

    for i, line in enumerate(lines):
        gen_texts = []
        if len(line.strip().split("\t")) != 4:
            prompt_wav, prompt_text, gen_wav, gen_text_1, gen_text_2, gen_text_3 = line.strip().split("\t")
            gen_texts = [gen_text_1, gen_text_2, gen_text_3]
        else:
            prompt_wav, prompt_text, gen_wav, gen_text = line.strip().split("\t")
            gen_texts = [gen_text]

        if prompt_wav.endswith(".mp4"):
            prompt_wav = prompt_wav.replace(".mp4", ".wav")
        
        wav_name = "_".join([gen_wav.split("/")[-2], gen_wav.split("/")[-1].split(".")[0]])
        save_path = f"{res_dir}/{wav_name}.wav"
        wav = video = None
        if gen_wav.endswith(".mp4"):
            video = gen_wav
        else:
            wav = gen_wav

        for i, gen_text in enumerate(gen_texts):
            if len(gen_texts) > 1:
                save_path = f"{res_dir}/{wav_name}_{i}.wav"

            metrics = generate_sentence(
                save_path=save_path,
                prompt_text=prompt_text,
                prompt_wav=prompt_wav,
                prompt_video=None,
                text=gen_text,
                wav=wav,
                video=video,
                model=model,
                vocoder=vocoder,
                tokenizer=tokenizer,
                feature_extractor=feature_extractor,
                device=device,
                num_step=num_step,
                video_guidance_scale=video_guidance_scale,
                face_guidance_scale=face_guidance_scale,
                lip_guidance_scale=lip_guidance_scale,
                guidance_scale=guidance_scale,
                speed=speed,
                t_shift=t_shift,
                target_rms=target_rms,
                feat_scale=feat_scale,
                sampling_rate=sampling_rate,
                params=params,
            )
        logging.info(f"[Sentence: {i}] RTF: {metrics['rtf']:.4f}")
        # total_t.append(metrics["t"])
        # total_t_no_vocoder.append(metrics["t_no_vocoder"])
        # total_t_vocoder.append(metrics["t_vocoder"])
        # total_wav_seconds.append(metrics["wav_seconds"])
        torch.cuda.empty_cache()

    # logging.info(f"Average RTF: {np.sum(total_t) / np.sum(total_wav_seconds):.4f}")
    # logging.info(
    #     f"Average RTF w/o vocoder: "
    #     f"{np.sum(total_t_no_vocoder) / np.sum(total_wav_seconds):.4f}"
    # )
    # logging.info(
    #     f"Average RTF vocoder: "
    #     f"{np.sum(total_t_vocoder) / np.sum(total_wav_seconds):.4f}"
    # )


@torch.inference_mode()
def main():
    parser = get_parser()
    args = parser.parse_args()

    params = AttributeDict()
    params.update(vars(args))
    fix_random_seed(params.seed)

    model_defaults = {
        "zipvoice": {
            "num_step": 16,
            "guidance_scale": 1.0,
        },
    }

    model_specific_defaults = model_defaults.get(params.model_name, {})

    for param, value in model_specific_defaults.items():
        if getattr(params, param) is None:
            setattr(params, param, value)
            logging.info(f"Setting {param} to default value: {value}")

    assert (params.test_list is not None) ^ (
        (params.prompt_wav and params.prompt_text and params.text) is not None
    ), (
        "For inference, please provide prompts and text with either '--test-list'"
        " or '--prompt-wav, --prompt-text and --text'."
    )

    params.model_dir = Path(params.model_dir)
    if not params.model_dir.is_dir():
        raise FileNotFoundError(f"{params.model_dir} does not exist")
    for filename in [params.checkpoint_name, "model.json", "tokens.txt"]:
        if not (params.model_dir / filename).is_file():
            raise FileNotFoundError(
                f"{params.model_dir / filename} does not exist. "
                "Download SyncVoice weights with: python tools/download_weights.py"
            )
    model_ckpt = params.model_dir / params.checkpoint_name
    model_config = params.model_dir / "model.json"
    token_file = params.model_dir / "tokens.txt"
    logging.info(
        f"Using local model dir {params.model_dir}, "
        f"checkpoint {params.checkpoint_name}"
    )

    logging.info("Loading model...")

    if params.tokenizer == "emilia":
        tokenizer = EmiliaTokenizer(token_file=token_file)
    elif params.tokenizer == "libritts":
        tokenizer = LibriTTSTokenizer(token_file=token_file)
    elif params.tokenizer == "espeak":
        tokenizer = EspeakTokenizer(token_file=token_file, lang=params.lang)
    else:
        assert params.tokenizer == "simple"
        tokenizer = SimpleTokenizer(token_file=token_file)

    tokenizer_config = {"vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id}

    with open(model_config, "r") as f:
        model_config = json.load(f)
        params.update(model_config["model"])
        params.update(model_config["feature"])

    model = ZipVoice(
        **model_config["model"],
        **tokenizer_config,
    )

    if str(model_ckpt).endswith(".safetensors"):
        safetensors.torch.load_model(model, model_ckpt)
    elif str(model_ckpt).endswith(".pt"):
        load_checkpoint(filename=model_ckpt, model=model, strict=False)
    else:
        raise NotImplementedError(f"Unsupported model checkpoint format: {model_ckpt}")

    if torch.cuda.is_available():
        params.device = torch.device("cuda", 0)
    elif torch.backends.mps.is_available():
        params.device = torch.device("mps")
    else:
        params.device = torch.device("cpu")
    logging.info(f"Device: {params.device}")

    model = model.to(params.device)
    model.eval()

    vocoder = get_vocoder(params.vocoder_path)
    vocoder = vocoder.to(params.device)
    vocoder.eval()

    if model_config["feature"]["type"] == "vocos":
        feature_extractor = VocosFbank()
    else:
        raise NotImplementedError(
            f"Unsupported feature type: {model_config['feature']['type']}"
        )
    params.sampling_rate = model_config["feature"]["sampling_rate"]

    logging.info("Start generating...")
    if params.test_list:
        os.makedirs(params.res_dir, exist_ok=True)
        generate_list(
            res_dir=params.res_dir,
            test_list=params.test_list,
            model=model,
            vocoder=vocoder,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            device=params.device,
            num_step=params.num_step,
            guidance_scale=params.guidance_scale,
            video_guidance_scale=params.video_guidance_scale,
            face_guidance_scale=params.face_guidance_scale,
            lip_guidance_scale=params.lip_guidance_scale,
            speed=params.speed,
            t_shift=params.t_shift,
            target_rms=params.target_rms,
            feat_scale=params.feat_scale,
            sampling_rate=params.sampling_rate,
            params=params,
        )
    else:
        generate_sentence(
            save_path=params.res_wav_path,
            prompt_text=params.prompt_text,
            prompt_wav=params.prompt_wav,
            prompt_video=params.prompt_video,
            text=params.text,
            wav=params.wav,
            video=params.video,
            model=model,
            vocoder=vocoder,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            device=params.device,
            num_step=params.num_step,
            guidance_scale=params.guidance_scale,
            video_guidance_scale=params.video_guidance_scale,
            face_guidance_scale=params.face_guidance_scale,
            lip_guidance_scale=params.lip_guidance_scale,
            speed=params.speed,
            t_shift=params.t_shift,
            target_rms=params.target_rms,
            feat_scale=params.feat_scale,
            sampling_rate=params.sampling_rate,
            params=params,
        )
    logging.info("Done")


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    main()
