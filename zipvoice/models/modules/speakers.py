#!/usr/bin/env python3
# Copyright 2025 Xiaomi Corp. / ZipVoice contributors
# Copyright 2026 SyncVoice contributors
#
# Speaker-conditioning blocks follow patterns from Tortoise TTS:
# https://github.com/neonbjb/tortoise-tts
#
# See NOTICE and THIRD_PARTY.md for attribution.
# Licensed under the Apache License, Version 2.0. See LICENSE.

import torch
import torch.nn as nn
import math
import logging


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    groups = 32
    if channels <= 16:
        groups = 8
    elif channels <= 64:
        groups = 16
    while channels % groups != 0:
        groups = int(groups / 2)
    assert groups > 2
    return GroupNorm32(groups, channels)


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/output heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, mask=None, rel_pos=None):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        if rel_pos is not None:
            weight = rel_pos(weight.reshape(bs, self.n_heads, weight.shape[-2], weight.shape[-1])).reshape(bs * self.n_heads, weight.shape[-2], weight.shape[-1])
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        if mask is not None:
            # The proper way to do this is to mask before the softmax using -inf, but that doesn't work properly on CPUs.
            mask = mask.repeat(self.n_heads, 1).unsqueeze(1)
            weight = weight * mask
        a = torch.einsum("bts,bcs->bct", weight, v)

        return a.reshape(bs, -1, length)


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        do_checkpoint=True,
        relative_pos_embeddings=False,
    ):
        super().__init__()
        self.channels = channels
        self.do_checkpoint = do_checkpoint
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        # split heads before split qkv
        self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))
        if relative_pos_embeddings:
            raise NotImplementedError
        else:
            self.relative_pos_embeddings = None

    def forward(self, x, mask=None):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv, mask, self.relative_pos_embeddings)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class ChannelNorm(nn.Module):
    """Normalize across channels for each sample and time frame.

    Input/output shape: ``[B, C, T]``.
    """
    def __init__(self, num_channels: int, affine: bool = True, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            # Match InstanceNorm1d affine parameter shapes for easy replacement.
            self.gamma = nn.Parameter(torch.ones(1, num_channels, 1))
            self.beta  = nn.Parameter(torch.zeros(1, num_channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)                  # [B, 1, T]
        var  = (x - mean).pow(2).mean(dim=1, keepdim=True)  # [B, 1, T]
        xhat = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            xhat = xhat * self.gamma + self.beta
        return xhat

class AttentionBlockQChannelNorm(nn.Module):
    """Multi-head self-attention with optional channel-wise Q/K normalization.

    Tensor layout: ``x`` has shape ``[B, D, T]``.
    """
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0,
                 q_cnorm: bool = True, k_cnorm: bool = False, use_ffn: bool = True):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 1x1 conv on [B, D, T] is equivalent to a per-channel linear projection.
        self.q_proj = nn.Conv1d(dim, dim, kernel_size=1)
        self.k_proj = nn.Conv1d(dim, dim, kernel_size=1)
        self.v_proj = nn.Conv1d(dim, dim, kernel_size=1)
        self.out_proj = nn.Conv1d(dim, dim, kernel_size=1)

        self.q_norm = ChannelNorm(dim, affine=True) if q_cnorm else nn.Identity()
        self.k_norm = ChannelNorm(dim, affine=True) if k_cnorm else nn.Identity()

        self.norm1 = nn.LayerNorm(dim)

        self.use_ffn = use_ffn
        if use_ffn:
            self.ffn = nn.Sequential(
                nn.Conv1d(dim, dim * 4, 1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(dim * 4, dim, 1),
                nn.Dropout(dropout),
            )
            self.norm2 = nn.LayerNorm(dim)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D, T] -> [B, H, d, T]
        B, D, T = x.shape
        x = x.view(B, self.num_heads, self.head_dim, T)
        return x

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, d, T] -> [B, D, T]
        B, H, d, T = x.shape
        return x.contiguous().view(B, H * d, T)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: ``[B, D, T]``
            attn_mask: optional ``[B, 1, T]`` or ``[B, T]``, broadcast to
                ``[B, H, T, T]``
        """
        residual = x

        q = self.q_proj(x)             # [B, D, T]
        k = self.k_proj(x)             # [B, D, T]
        v = self.v_proj(x)             # [B, D, T]

        q = self.q_norm(q)             # [B, D, T]
        k = self.k_norm(k)             # [B, D, T]

        q = self._split_heads(q)       # [B, H, d, T]
        k = self._split_heads(k)       # [B, H, d, T]
        v = self._split_heads(v)       # [B, H, d, T]

        q = q.permute(0, 1, 3, 2)  # [B, H, T, d]
        k = k.permute(0, 1, 3, 2)  # [B, H, T, d]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, T, T]

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask[:, None, None, :]  # [B, 1, 1, T]
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask[:, None, :, :]     # [B, 1, T, T]

            attn = attn.masked_fill(attn_mask == 0, float("-inf"))

        attn = attn.softmax(dim=-1)

        v = v.permute(0, 1, 3, 2)  # [B, H, T, d]
        out = torch.matmul(attn, v)  # [B, H, T, d]
        out = out.permute(0, 1, 3, 2)  # [B, H, d, T]

        out = self._merge_heads(out)   # [B, D, T]
        out = self.out_proj(out)

        x = residual + out
        x = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B, D, T]

        if self.use_ffn:
            residual_ffn = x
            x = self.ffn(x)
            x = residual_ffn + x
            x = self.norm2(x.permute(0, 2, 1)).permute(0, 2, 1)

        return x


class ConditioningEncoder(nn.Module):
    def __init__(self,
                 spec_dim,
                 model_dim,
                 embedding_dim,
                 attn_blocks=6,
                 num_attn_heads=4,
                 do_checkpointing=False,
                 mean=False):
        super().__init__()
        attn = []
        logging.info("Using Tortoise-style speaker conditioning encoder")
        self.init = nn.Conv1d(spec_dim, model_dim, kernel_size=1)
        for a in range(attn_blocks):
            attn.append(AttentionBlock(model_dim, num_attn_heads))
        self.attn = nn.Sequential(*attn)
        self.dim = embedding_dim
        self.do_checkpointing = do_checkpointing
        self.mean = mean
        self.fc = nn.Linear(model_dim, embedding_dim)

    def forward(self, x, attn_mask=None):
        h = self.init(x)
        for block in self.attn:
            h = block(h, attn_mask.squeeze(1))

        if self.mean:
            return self.fc(h.mean(dim=2))
        else:
            return self.fc(h[:, :, 0])

class SpeakerEncoder(nn.Module):
    def __init__(self,
                 spec_dim: int,
                 model_dim: int,
                 embedding_dim: int,
                 attn_blocks: int = 6,
                 num_attn_heads: int = 4,
                 q_cnorm: bool = True,
                 k_cnorm: bool = False,
                 use_ffn: bool = False):
        super().__init__()

        self.init = nn.Conv1d(spec_dim, model_dim, kernel_size=1)
        # self.init = nn.Conv1d(spec_dim, model_dim, kernel_size=3, padding=1)
    
        blocks = []
        for _ in range(attn_blocks):
            blocks.append(
                AttentionBlockQChannelNorm(
                    dim=model_dim,
                    num_heads=num_attn_heads,
                    dropout=0.1,
                    q_cnorm=q_cnorm,
                    k_cnorm=k_cnorm,
                    use_ffn=use_ffn,
                )
            )
        self.attn = nn.Sequential(*blocks)

        self.fc = nn.Linear(model_dim * 2, embedding_dim)

    def forward(self, x, attn_mask=None):
        """
        Args:
            x: ``[B, spec_dim, T]``
            attn_mask: ``[B, 1, T]``
        Returns:
            Speaker embedding of shape ``[B, embedding_dim]``.
        """
        h = self.init(x)          # [B, model_dim, T]
        for block in self.attn:
            h = block(h, attn_mask)
        mean = h.mean(dim=2)      # [B, model_dim]
        std  = h.std(dim=2)       # [B, model_dim]
        pooled = torch.cat([mean, std], dim=1)  # [B, 2 * model_dim]

        out = self.fc(pooled)     # [B, D]
        return out
