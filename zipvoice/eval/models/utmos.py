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

"""UTMOS strong model adapted from https://github.com/tarepan/SpeechMOS."""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class UTMOS22Strong(nn.Module):
    """Saeki et al. (2022) UTMOS strong learner without phoneme encoder."""

    def __init__(self):
        super().__init__()
        feat_ssl, feat_domain_emb, feat_judge_emb = 768, 128, 128
        feat_rnn_h, feat_proj_h = 512, 2048
        feat_cat = feat_ssl + feat_domain_emb + feat_judge_emb
        self.wav2vec2 = Wav2Vec2Model()
        self.domain_emb = nn.Parameter(
            torch.empty(1, feat_domain_emb), requires_grad=False
        )
        self.judge_emb = nn.Parameter(
            torch.empty(1, feat_judge_emb), requires_grad=False
        )
        self.blstm = nn.LSTM(
            feat_cat, feat_rnn_h, batch_first=True, bidirectional=True
        )
        self.projection = nn.Sequential(
            nn.Linear(feat_rnn_h * 2, feat_proj_h),
            nn.ReLU(),
            nn.Linear(feat_proj_h, 1),
        )

    def forward(self, wave: Tensor, sr: int) -> Tensor:
        del sr
        unit_series = self.wav2vec2(wave)
        bsz, frames, _ = unit_series.size()
        domain_series = self.domain_emb.unsqueeze(1).expand(bsz, frames, -1)
        judge_series = self.judge_emb.unsqueeze(1).expand(bsz, frames, -1)
        features = torch.cat(
            [unit_series, domain_series, judge_series], dim=2
        )
        scores = self.projection(self.blstm(features)[0])
        return scores.mean(dim=1).squeeze(1) * 2 + 3


class Wav2Vec2Model(nn.Module):
    def __init__(self):
        super().__init__()
        feature_layers = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [
            (512, 2, 2)
        ] * 2
        self.feature_extractor = ConvFeatureExtractionModel(feature_layers)
        self.layer_norm = nn.LayerNorm(512)
        self.post_extract_proj = nn.Linear(512, 768)
        self.dropout_input = nn.Dropout(0.1)
        self.encoder = TransformerEncoder(768)
        self.mask_emb = nn.Parameter(torch.FloatTensor(768))

    def forward(self, source: Tensor) -> Tensor:
        features = self.feature_extractor(source).transpose(1, 2)
        features = self.post_extract_proj(self.layer_norm(features))
        return self.encoder(features)


class ConvFeatureExtractionModel(nn.Module):
    def __init__(self, conv_layers: List[Tuple[int, int, int]]):
        super().__init__()

        def block(n_in, n_out, kernel, stride, group_norm=False):
            layers = [
                nn.Conv1d(
                    n_in, n_out, kernel, stride=stride, bias=False
                ),
                nn.Dropout(p=0.0),
            ]
            if group_norm:
                layers.append(nn.GroupNorm(n_out, n_out, affine=True))
            layers.append(nn.GELU())
            return nn.Sequential(*layers)

        in_dim = 1
        self.conv_layers = nn.ModuleList()
        for index, (dim, kernel, stride) in enumerate(conv_layers):
            self.conv_layers.append(
                block(
                    in_dim,
                    dim,
                    kernel,
                    stride,
                    group_norm=index == 0,
                )
            )
            in_dim = dim

    def forward(self, series: Tensor) -> Tensor:
        series = series.unsqueeze(1)
        for conv in self.conv_layers:
            series = conv(series)
        return series


class TransformerEncoder(nn.Module):
    def __init__(self, feat: int):
        super().__init__()
        self.required_seq_len_multiple = 2
        self.pos_conv = nn.Sequential(
            nn.utils.weight_norm(
                nn.Conv1d(
                    feat,
                    feat,
                    kernel_size=128,
                    padding=64,
                    groups=16,
                ),
                name="weight",
                dim=2,
            ),
            SamePad(128),
            nn.GELU(),
        )
        self.layer_norm = nn.LayerNorm(feat)
        self.layers = nn.ModuleList(
            [self.build_encoder_layer(feat) for _ in range(12)]
        )

    @staticmethod
    def build_encoder_layer(feat: int):
        return TransformerSentenceEncoderLayer(
            embedding_dim=feat,
            ffn_embedding_dim=3072,
            num_attention_heads=12,
            activation_fn="gelu",
            dropout=0.1,
            attention_dropout=0.1,
            activation_dropout=0.0,
            layer_norm_first=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pos_conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.layer_norm(x)
        x, pad_length = pad_to_multiple(
            x, self.required_seq_len_multiple, dim=-2, value=0
        )
        if pad_length > 0:
            padding_mask = x.new_zeros(
                (x.size(0), x.size(1)), dtype=torch.bool
            )
            padding_mask[:, -pad_length:] = True
        else:
            padding_mask = None
        x = x.transpose(0, 1)
        for layer in self.layers:
            x = layer(x, padding_mask)
        x = x.transpose(0, 1)
        return x[:, :-pad_length] if pad_length > 0 else x


class SamePad(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        if kernel_size % 2:
            raise ValueError("SamePad supports only even kernels")

    def forward(self, x: Tensor) -> Tensor:
        return x[:, :, :-1]


def pad_to_multiple(
    x: Optional[Tensor],
    multiple: int,
    dim: int = -1,
    value: float = 0,
) -> Tuple[Optional[Tensor], int]:
    if x is None:
        return None, 0
    size = x.size(dim)
    remainder = math.ceil(size / multiple) * multiple - size
    if remainder == 0:
        return x, 0
    pad_offset = (0,) * (-1 - dim) * 2
    return (
        F.pad(x, (*pad_offset, 0, remainder), value=value),
        remainder,
    )


class TransformerSentenceEncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        ffn_embedding_dim: int,
        num_attention_heads: int,
        activation_fn: str,
        dropout: float,
        attention_dropout: float,
        activation_dropout: float,
        layer_norm_first: bool,
    ):
        super().__init__()
        if layer_norm_first or activation_fn != "gelu":
            raise ValueError("Only post-norm GELU encoder layers are supported")
        self.self_attn = MultiheadAttention(
            embedding_dim, num_attention_heads, attention_dropout
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(activation_dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, embedding_dim)
        self.self_attn_layer_norm = nn.LayerNorm(embedding_dim)
        self.final_layer_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self, x: Tensor, self_attn_padding_mask: Optional[Tensor]
    ) -> Tensor:
        residual = x
        x = self.dropout1(
            self.self_attn(x, x, x, self_attn_padding_mask)
        )
        x = self.self_attn_layer_norm(residual + x)
        residual = x
        x = self.fc2(self.dropout2(F.gelu(self.fc1(x))))
        x = self.dropout3(x)
        return self.final_layer_norm(residual + x)


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.p_dropout = dropout
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        return F.multi_head_attention_forward(
            query=query,
            key=key,
            value=value,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads,
            in_proj_weight=torch.empty([0]),
            in_proj_bias=torch.cat(
                (self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=self.p_dropout,
            out_proj_weight=self.out_proj.weight,
            out_proj_bias=self.out_proj.bias,
            training=False,
            key_padding_mask=(
                key_padding_mask.bool()
                if key_padding_mask is not None
                else None
            ),
            need_weights=False,
            use_separate_proj_weight=True,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
        )[0]
