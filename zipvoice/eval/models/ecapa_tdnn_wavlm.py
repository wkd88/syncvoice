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

"""WavLM-based ECAPA-TDNN speaker encoder.

Architecture adapted from https://github.com/lawlict/ECAPA-TDNN with a WavLM
frontend via https://github.com/s3prl/s3prl.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class ECAPA_TDNN_WAVLM(nn.Module):
    def __init__(
        self,
        feat_dim=80,
        channels=512,
        emb_dim=192,
        global_context_att=False,
        sr=16000,
        ssl_model_path=None,
    ):
        super().__init__()
        self.sr = sr
        if ssl_model_path is None:
            self.feature_extract = torch.hub.load(
                "s3prl/s3prl", "wavlm_large"
            )
        else:
            self.feature_extract = torch.hub.load(
                ssl_model_path,
                "wavlm_local",
                source="local",
                ckpt=os.path.join(ssl_model_path, "wavlm_large.pt"),
            )
        layers = self.feature_extract.model.encoder.layers
        for index in (23, 11):
            if len(layers) > index and hasattr(
                layers[index].self_attn, "fp32_attention"
            ):
                layers[index].self_attn.fp32_attention = False

        self.feat_num = self.get_feat_num()
        self.feature_weight = nn.Parameter(torch.zeros(self.feat_num))
        self.instance_norm = nn.InstanceNorm1d(feat_dim)
        self.channels = [channels] * 4 + [1536]
        self.layer1 = Conv1dReluBn(
            feat_dim, self.channels[0], kernel_size=5, padding=2
        )
        self.layer2 = SE_Res2Block(
            self.channels[0], self.channels[1], 3, 1, 2, 2, 8, 128
        )
        self.layer3 = SE_Res2Block(
            self.channels[1], self.channels[2], 3, 1, 3, 3, 8, 128
        )
        self.layer4 = SE_Res2Block(
            self.channels[2], self.channels[3], 3, 1, 4, 4, 8, 128
        )
        self.conv = nn.Conv1d(channels * 3, self.channels[-1], 1)
        self.pooling = AttentiveStatsPool(
            self.channels[-1],
            attention_channels=128,
            global_context_att=global_context_att,
        )
        self.bn = nn.BatchNorm1d(self.channels[-1] * 2)
        self.linear = nn.Linear(self.channels[-1] * 2, emb_dim)

    def get_feat_num(self):
        self.feature_extract.eval()
        device = next(self.feature_extract.parameters()).device
        wav = [torch.randn(self.sr, device=device)]
        with torch.no_grad():
            features = self.feature_extract(wav)["hidden_states"]
        return len(features) if isinstance(features, (list, tuple)) else 1

    def get_feat(self, x):
        with torch.no_grad():
            x = self.feature_extract([sample for sample in x])
        x = x["hidden_states"]
        x = torch.stack(x, dim=0) if isinstance(
            x, (list, tuple)
        ) else x.unsqueeze(0)
        weights = F.softmax(self.feature_weight, dim=-1)
        weights = weights[:, None, None, None]
        x = (weights * x).sum(dim=0).transpose(1, 2) + 1e-6
        return self.instance_norm(x)

    def forward(self, x):
        out1 = self.layer1(self.get_feat(x))
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = F.relu(self.conv(torch.cat([out2, out3, out4], dim=1)))
        return self.linear(self.bn(self.pooling(out)))


class Res2Conv1dReluBn(nn.Module):
    """Res2Net-style grouped convolution."""

    def __init__(
        self,
        channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        scale=4,
    ):
        super().__init__()
        if channels % scale != 0:
            raise ValueError(f"{channels} is not divisible by {scale}")
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    self.width,
                    self.width,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    bias=bias,
                )
                for _ in range(self.nums)
            ]
        )
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(self.width) for _ in range(self.nums)]
        )

    def forward(self, x):
        outputs = []
        splits = torch.split(x, self.width, dim=1)
        current = None
        for index in range(self.nums):
            current = (
                splits[index]
                if index == 0
                else current + splits[index]
            )
            current = self.bns[index](F.relu(self.convs[index](current)))
            outputs.append(current)
        if self.scale != 1:
            outputs.append(splits[self.nums])
        return torch.cat(outputs, dim=1)


class Conv1dReluBn(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            bias=bias,
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


class SE_Connect(nn.Module):
    def __init__(self, channels, se_bottleneck_dim=128):
        super().__init__()
        self.linear1 = nn.Linear(channels, se_bottleneck_dim)
        self.linear2 = nn.Linear(se_bottleneck_dim, channels)

    def forward(self, x):
        weights = F.relu(self.linear1(x.mean(dim=2)))
        weights = torch.sigmoid(self.linear2(weights))
        return x * weights.unsqueeze(2)


class SE_Res2Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        scale,
        se_bottleneck_dim,
    ):
        super().__init__()
        self.Conv1dReluBn1 = Conv1dReluBn(in_channels, out_channels)
        self.Res2Conv1dReluBn = Res2Conv1dReluBn(
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            scale=scale,
        )
        self.Conv1dReluBn2 = Conv1dReluBn(out_channels, out_channels)
        self.SE_Connect = SE_Connect(out_channels, se_bottleneck_dim)
        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x):
        residual = self.shortcut(x) if self.shortcut is not None else x
        x = self.Conv1dReluBn1(x)
        x = self.Res2Conv1dReluBn(x)
        x = self.Conv1dReluBn2(x)
        return self.SE_Connect(x) + residual


class AttentiveStatsPool(nn.Module):
    def __init__(
        self, in_dim, attention_channels=128, global_context_att=False
    ):
        super().__init__()
        self.global_context_att = global_context_att
        input_dim = in_dim * 3 if global_context_att else in_dim
        self.linear1 = nn.Conv1d(input_dim, attention_channels, 1)
        self.linear2 = nn.Conv1d(attention_channels, in_dim, 1)

    def forward(self, x):
        if self.global_context_att:
            context_mean = x.mean(dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(
                torch.var(x, dim=-1, keepdim=True) + 1e-10
            ).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x
        alpha = torch.softmax(
            self.linear2(torch.tanh(self.linear1(x_in))), dim=2
        )
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * (x**2), dim=2) - mean**2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)
