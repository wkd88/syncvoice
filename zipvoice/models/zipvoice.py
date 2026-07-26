# Copyright    2024    Xiaomi Corp.        (authors:  Wei Kang
#                                                     Han Zhu)
#
# See ../../../../LICENSE for clarification regarding multiple authors
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

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from zipvoice.models.modules.speakers import SpeakerEncoder, ConditioningEncoder
from zipvoice.models.modules.solver import EulerSolver
from zipvoice.models.modules.zipformer import TTSZipformer
from zipvoice.utils.common import (
    get_tokens_index,
    make_pad_mask,
    pad_labels,
    prepare_avg_tokens_durations,
)

class Adapter(nn.Module):
    def __init__(self, d_model, out_dim, reduction_ratio=4):
        super().__init__()
        self.down_proj = nn.Linear(d_model, d_model // reduction_ratio)
        self.up_proj = nn.Linear(d_model // reduction_ratio, d_model)
        self.activation = nn.GELU()
        self.final_linear = nn.Linear(d_model, out_dim)
        self.input_norm = nn.LayerNorm(d_model) 
        self.residual_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.input_norm(x)
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.up_proj(x)
        x = self.residual_norm(x + residual)
        output = self.final_linear(x)
        return output

class TextVisualFusion(nn.Module):
    def __init__(self, lip_dim=512, face_dim=256, hidden_dim=100):
        super().__init__()
        self.lip_adapter = Adapter(lip_dim, hidden_dim)
        self.face_adapter = Adapter(face_dim, hidden_dim)
        self.lip_fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.face_fusion = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, text_feat, face_feat, lip_feat):
        face_feat = self.face_adapter(face_feat)
        lip_feat = self.lip_adapter(lip_feat)
        tf_feat = torch.cat([text_feat, face_feat], dim=2)
        tl_feat = torch.cat([text_feat, lip_feat], dim=2)
        tf_feat = self.face_fusion(tf_feat)
        tl_feat = self.lip_fusion(tl_feat)
        return text_feat + tf_feat + tl_feat



class ZipVoice(nn.Module):
    """The ZipVoice model."""

    def __init__(
        self,
        fm_decoder_downsampling_factor: List[int] = [1, 2, 4, 2, 1],
        fm_decoder_num_layers: List[int] = [2, 2, 4, 4, 4],
        fm_decoder_cnn_module_kernel: List[int] = [31, 15, 7, 15, 31],
        fm_decoder_feedforward_dim: int = 1536,
        fm_decoder_num_heads: int = 4,
        fm_decoder_dim: int = 512,
        text_encoder_num_layers: int = 4,
        text_encoder_feedforward_dim: int = 512,
        text_encoder_cnn_module_kernel: int = 9,
        text_encoder_num_heads: int = 4,
        text_encoder_dim: int = 192,
        time_embed_dim: int = 192,
        text_embed_dim: int = 192,
        query_head_dim: int = 32,
        value_head_dim: int = 12,
        pos_head_dim: int = 4,
        pos_dim: int = 48,
        feat_dim: int = 100,
        vocab_size: int = 26,
        pad_id: int = 0,
        use_spk: bool = False,
        spk_encoder_dim: int = 256,
        spk_embed_dim: int = 192,
        use_pretrain_spk: bool = False,
        spk_encoder_q_cnorm: bool = True,
        tortoise_encoder: bool = False,
        use_spk_encoder_pretrain: bool = False,
    ):
        """
        Initialize the model with specified configuration parameters.

        Args:
            fm_decoder_downsampling_factor: List of downsampling factors for each layer
                in the flow-matching decoder.
            fm_decoder_num_layers: List of the number of layers for each block in the
                flow-matching decoder.
            fm_decoder_cnn_module_kernel: List of kernel sizes for CNN modules in the
                flow-matching decoder.
            fm_decoder_feedforward_dim: Dimension of the feedforward network in the
                flow-matching decoder.
            fm_decoder_num_heads: Number of attention heads in the flow-matching
                decoder.
            fm_decoder_dim: Hidden dimension of the flow-matching decoder.
            text_encoder_num_layers: Number of layers in the text encoder.
            text_encoder_feedforward_dim: Dimension of the feedforward network in the
                text encoder.
            text_encoder_cnn_module_kernel: Kernel size for the CNN module in the
                text encoder.
            text_encoder_num_heads: Number of attention heads in the text encoder.
            text_encoder_dim: Hidden dimension of the text encoder.
            time_embed_dim: Dimension of the time embedding.
            text_embed_dim: Dimension of the text embedding.
            query_head_dim: Dimension of the query attention head.
            value_head_dim: Dimension of the value attention head.
            pos_head_dim: Dimension of the position attention head.
            pos_dim: Dimension of the positional encoding.
            feat_dim: Dimension of the acoustic features.
            vocab_size: Size of the vocabulary.
            pad_id: ID used for padding tokens.
        """
        super().__init__()
        self.use_spk = use_spk
        self.use_pretrain_spk = use_pretrain_spk
        self.use_spk_encoder_pretrain = use_spk_encoder_pretrain

        if use_spk and not use_pretrain_spk:
            if tortoise_encoder:
                self.speaker_encoder = ConditioningEncoder(feat_dim, spk_encoder_dim, spk_embed_dim)
            else:
                self.speaker_encoder = SpeakerEncoder(feat_dim, spk_encoder_dim, spk_embed_dim, q_cnorm=spk_encoder_q_cnorm)

            if not use_spk_encoder_pretrain:
                self.map = nn.Sequential(
                    nn.Linear(spk_embed_dim, spk_embed_dim * 2),
                    nn.ReLU(),
                    nn.Linear(spk_embed_dim * 2, spk_embed_dim)
                )
            # raise NotImplementedError

        self.fm_decoder = TTSZipformer(
            in_dim=feat_dim * 3,
            out_dim=feat_dim,
            downsampling_factor=fm_decoder_downsampling_factor,
            num_encoder_layers=fm_decoder_num_layers,
            cnn_module_kernel=fm_decoder_cnn_module_kernel,
            encoder_dim=fm_decoder_dim,
            feedforward_dim=fm_decoder_feedforward_dim,
            num_heads=fm_decoder_num_heads,
            query_head_dim=query_head_dim,
            pos_head_dim=pos_head_dim,
            value_head_dim=value_head_dim,
            pos_dim=pos_dim,
            use_time_embed=True,
            time_embed_dim=time_embed_dim,
            use_spk_embed=use_spk,
            spk_embed_dim=spk_embed_dim
        )

        self.text_encoder = TTSZipformer(
            in_dim=text_embed_dim,
            out_dim=feat_dim,
            downsampling_factor=1,
            num_encoder_layers=text_encoder_num_layers,
            cnn_module_kernel=text_encoder_cnn_module_kernel,
            encoder_dim=text_encoder_dim,
            feedforward_dim=text_encoder_feedforward_dim,
            num_heads=text_encoder_num_heads,
            query_head_dim=query_head_dim,
            pos_head_dim=pos_head_dim,
            value_head_dim=value_head_dim,
            pos_dim=pos_dim,
            use_time_embed=False,
        )

        self.tv_fusion = TextVisualFusion(lip_dim=512, face_dim=256, hidden_dim=feat_dim)
        # self.visual_linear = nn.Linear(256+512, feat_dim)
        # self.cond_linear = nn.Linear(feat_dim * 2, feat_dim)

        self.feat_dim = feat_dim
        self.text_embed_dim = text_embed_dim
        self.pad_id = pad_id

        self.embed = nn.Embedding(vocab_size, text_embed_dim)
        self.solver = EulerSolver(self, func_name="forward_fm_decoder")

    def forward_fm_decoder(
        self,
        t: torch.Tensor,
        xt: torch.Tensor,
        text_condition: torch.Tensor,
        speech_condition: torch.Tensor,
        speaker_condition: torch.Tensor = None,
        padding_mask: Optional[torch.Tensor] = None,
        guidance_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute velocity.
        Args:
            t:  A tensor of shape (N, 1, 1) or a tensor of a float,
                in the range of (0, 1).
            xt: the input of the current timestep, including condition
                embeddings and noisy acoustic features.
            text_condition: the text condition embeddings, with the
                shape (batch, seq_len, emb_dim).
            speech_condition: the speech condition embeddings, with the
                shape (batch, seq_len, emb_dim).
            speaker_condition: the speaker condition embeddings, with the
                shape (batch, emb_dim). 
            padding_mask: The mask for padding, True means masked
                position, with the shape (N, T).
            guidance_scale: The guidance scale in classifier-free guidance,
                which is a tensor of shape (N, 1, 1) or a tensor of a float.

        Returns:
            predicted velocity, with the shape (batch, seq_len, emb_dim).
        """

        xt = torch.cat([xt, text_condition, speech_condition], dim=2)

        assert t.dim() in (0, 3)
        # Handle t with the shape (N, 1, 1):
        # squeeze the last dimension if it's size is 1.
        while t.dim() > 1 and t.size(-1) == 1:
            t = t.squeeze(-1)
        # Handle t with a single value: expand to the size of batch size.
        if t.dim() == 0:
            t = t.repeat(xt.shape[0])

        if guidance_scale is not None:
            while guidance_scale.dim() > 1 and guidance_scale.size(-1) == 1:
                guidance_scale = guidance_scale.squeeze(-1)
            if guidance_scale.dim() == 0:
                guidance_scale = guidance_scale.repeat(xt.shape[0])

            vt = self.fm_decoder(
                x=xt, t=t, speaker_condition=speaker_condition, padding_mask=padding_mask, guidance_scale=guidance_scale
            )
        else:
            vt = self.fm_decoder(x=xt, t=t, speaker_condition=speaker_condition, padding_mask=padding_mask)
        return vt

    def forward_text_embed(
        self,
        tokens: List[List[int]],
    ):
        """
        Get the text embeddings.
        Args:
            tokens: a list of list of token ids.
        Returns:
            embed: the text embeddings, shape (batch, seq_len, emb_dim).
            tokens_lens: the length of each token sequence, shape (batch,).
        """
        device = (
            self.device if isinstance(self, DDP) else next(self.parameters()).device
        )
        tokens_padded = pad_labels(tokens, pad_id=self.pad_id, device=device)  # (B, S)
        embed = self.embed(tokens_padded)  # (B, S, C)
        tokens_lens = torch.tensor(
            [len(token) for token in tokens], dtype=torch.int64, device=device
        )
        tokens_padding_mask = make_pad_mask(tokens_lens, embed.shape[1])  # (B, S)

        embed = self.text_encoder(
            x=embed, t=None, padding_mask=tokens_padding_mask
        )  # (B, S, C)
        return embed, tokens_lens

    def forward_text_condition(
        self,
        embed: torch.Tensor,
        tokens_lens: torch.Tensor,
        features_lens: torch.Tensor,
    ):
        """
        Get the text condition with the same length of the acoustic feature.
        Args:
            embed: the text embeddings, shape (batch, token_seq_len, emb_dim).
            tokens_lens: the length of each token sequence, shape (batch,).
            features_lens: the length of each acoustic feature sequence,
                shape (batch,).
        Returns:
            text_condition: the text condition, shape
                (batch, feature_seq_len, emb_dim).
            padding_mask: the padding mask of text condition, shape
                (batch, feature_seq_len).
        """

        num_frames = int(features_lens.max())

        padding_mask = make_pad_mask(features_lens, max_len=num_frames)  # (B, T)

        tokens_durations = prepare_avg_tokens_durations(features_lens, tokens_lens)

        tokens_index = get_tokens_index(tokens_durations, num_frames).to(
            embed.device
        )  # (B, T)

        text_condition = torch.gather(
            embed,
            dim=1,
            index=tokens_index.unsqueeze(-1).expand(
                embed.size(0), num_frames, embed.size(-1)
            ),
        )  # (B, T, F)
        return text_condition, padding_mask

    def forward_text_inference_gt_duration(
        self,
        tokens: List[List[int]],
        features_lens: torch.Tensor,
        prompt_tokens: List[List[int]],
        prompt_features_lens: torch.Tensor,
    ):
        """
        Process text for inference, given text tokens, real feature lengths and prompts.
        """
        if prompt_tokens is None:
            tokens = [token for token in tokens]
            features_lens = features_lens
        else:
            tokens = [
                prompt_token + token for prompt_token, token in zip(prompt_tokens, tokens)
            ]
            features_lens = prompt_features_lens + features_lens

        embed, tokens_lens = self.forward_text_embed(tokens)
        text_condition, padding_mask = self.forward_text_condition(
            embed, tokens_lens, features_lens
        )
        return text_condition, padding_mask

    def forward_text_inference_ratio_duration(
        self,
        tokens: List[List[int]],
        prompt_tokens: List[List[int]],
        prompt_features_lens: torch.Tensor,
        speed: float,
    ):
        """
        Process text for inference, given text tokens and prompts,
        feature lengths are predicted with the ratio of token numbers.
        """
        device = (
            self.device if isinstance(self, DDP) else next(self.parameters()).device
        )

        cat_tokens = [
            prompt_token + token for prompt_token, token in zip(prompt_tokens, tokens)
        ]

        prompt_tokens_lens = torch.tensor(
            [len(token) for token in prompt_tokens],
            dtype=torch.int64,
            device=device,
        )

        tokens_lens = torch.tensor(
            [len(token) for token in tokens],
            dtype=torch.int64,
            device=device,
        )

        cat_embed, cat_tokens_lens = self.forward_text_embed(cat_tokens)

        features_lens = prompt_features_lens + torch.ceil(
            (prompt_features_lens / prompt_tokens_lens * tokens_lens / speed)
        ).to(dtype=torch.int64)

        text_condition, padding_mask = self.forward_text_condition(
            cat_embed, cat_tokens_lens, features_lens
        )
        return text_condition, padding_mask

    def get_speaker_conditioning(self, reference_mels, reference_mel_masks):
        """Extract speaker conditioning from reference audio or embeddings."""
        # Handle multiple references
        B, N, T, C = reference_mels.shape
        conds = []
        for i in range(N):
            ref_mel = reference_mels[:, i, :, :]
            if reference_mel_masks is not None:
                mask = reference_mel_masks[:, i, :].unsqueeze(1)
            else:
                mask = None

            cond = self.speaker_encoder(ref_mel.transpose(1, 2), mask)  # [B, spk_embed_dim]
            conds.append(cond)
        # Average multiple references (Tortoise-style aggregation)
        speaker_embed = torch.stack(conds, dim=1).mean(dim=1)  # [B, spk_embed_dim]

        speaker_embed = F.normalize(speaker_embed, dim=1)
        # speaker_embed = speaker_embed.unsqueeze(1)  # [B, 1, llm_input_size]
            
        return speaker_embed

    def sample(
        self,
        tokens: List[List[int]],
        prompt_tokens: List[List[int]],
        prompt_features: torch.Tensor,
        prompt_features_lens: torch.Tensor,
        features_lens: Optional[torch.Tensor] = None,
        lip_features: Optional[torch.Tensor] = None,
        face_features: Optional[torch.Tensor] = None,
        spk_features: Optional[torch.Tensor] = None,
        reference_mels: Optional[torch.Tensor] = None,
        reference_mel_masks: Optional[torch.Tensor] = None,
        speed: float = 1.0,
        t_shift: float = 1.0,
        duration: str = "predict",
        num_step: int = 5,
        guidance_scale: float = 0.5,
        video_guidance_scale: float = 0.5,
        lip_guidance_scale: float = 0.5,
        face_guidance_scale: float = 0.5,
    ) -> torch.Tensor:
        """
        Generate acoustic features, given text tokens, prompts feature
            and prompt transcription's text tokens.
        Args:
            tokens: a list of list of text tokens.
            prompt_tokens: a list of list of prompt tokens.
            prompt_features: the prompt feature with the shape
                (batch_size, seq_len, feat_dim).
            prompt_features_lens: the length of each prompt feature,
                with the shape (batch_size,).
            features_lens: the length of the predicted eature, with the
                shape (batch_size,). It is used only when duration is "real".
            duration: "real" or "predict". If "real", the predicted
                feature length is given by features_lens.
            num_step: the number of steps to use in the ODE solver.
            guidance_scale: the guidance scale for classifier-free guidance.
        """

        assert duration in ["real", "predict"]
        if prompt_features is None:
            assert duration == "real"

        if duration == "predict":
            (
                text_condition,
                padding_mask,
            ) = self.forward_text_inference_ratio_duration(
                tokens=tokens,
                prompt_tokens=prompt_tokens,
                prompt_features_lens=prompt_features_lens,
                speed=speed,
            )
        else:
            assert features_lens is not None
            text_condition, padding_mask = self.forward_text_inference_gt_duration(
                tokens=tokens,
                features_lens=features_lens,
                prompt_tokens=prompt_tokens,
                prompt_features_lens=prompt_features_lens,
            )
        batch_size, num_frames, _ = text_condition.shape
  
        if self.use_spk and (spk_features is not None or reference_mels is not None):
            if self.use_pretrain_spk:
                spk_condition = spk_features
            else:
                spk_condition = self.get_speaker_conditioning(reference_mels, reference_mel_masks)
                if self.use_spk_encoder_pretrain:
                    spk_condition = spk_condition + spk_features
                else:
                    spk_predict = self.map(spk_condition)
                    spk_predict = F.normalize(spk_predict, dim=1)
                    # spk_condition = spk_predict
        else:
            spk_condition = None

        if lip_features is None:
            face_features = torch.zeros((batch_size, num_frames, 256), device=text_condition.device)
            lip_features = torch.zeros((batch_size, num_frames, 512), device=text_condition.device)
        # text_condition = torch.cat([text_condition, visual_condition], dim=2)

        if lip_guidance_scale != -5.0 or face_guidance_scale != -5.0:
            if lip_guidance_scale != -5.0 and face_guidance_scale != -5.0:
                video_text_condition = self.tv_fusion(text_condition, face_features, lip_features)
                lip_text_condition = self.tv_fusion(text_condition, torch.zeros_like(face_features), lip_features)
                face_text_condition = self.tv_fusion(text_condition, face_features, torch.zeros_like(lip_features))
            elif lip_guidance_scale != -5.0:
                video_text_condition = self.tv_fusion(text_condition, torch.zeros_like(face_features), lip_features)
                lip_text_condition = self.tv_fusion(text_condition, torch.zeros_like(face_features), lip_features)
                face_text_condition = None
            elif face_guidance_scale != -5.0:
                video_text_condition = self.tv_fusion(text_condition, face_features, torch.zeros_like(lip_features))
                lip_text_condition = None
                face_text_condition = self.tv_fusion(text_condition, face_features, torch.zeros_like(lip_features))  

            text_condition = self.tv_fusion(text_condition, torch.zeros_like(face_features), torch.zeros_like(lip_features)) 
        else:
            video_text_condition = None
            lip_text_condition = None
            face_text_condition = None
            text_condition = self.tv_fusion(text_condition, torch.zeros_like(face_features), torch.zeros_like(lip_features))

        if prompt_features is None:
            speech_condition = torch.zeros(
                batch_size, num_frames, 100, device=text_condition.device
            )
        else:            
            speech_condition = torch.nn.functional.pad(
                prompt_features, (0, 0, 0, num_frames - prompt_features.size(1))
            )  # (B, T, F)

        if prompt_features is None:
            prompt_features_lens = torch.zeros(
                batch_size, dtype=torch.int64, device=text_condition.device
            )

        # False means speech condition positions.
        speech_condition_mask = make_pad_mask(prompt_features_lens, num_frames)
        speech_condition = torch.where(
            speech_condition_mask.unsqueeze(-1),
            torch.zeros_like(speech_condition),
            speech_condition,
        )

        x0 = torch.randn(
            batch_size,
            num_frames,
            100, # prompt_features.size(-1),
            device=text_condition.device,
        )

        x1 = self.solver.sample(
            x=x0,
            text_condition=text_condition,
            speech_condition=speech_condition,
            padding_mask=padding_mask,
            num_step=num_step,
            guidance_scale=guidance_scale,
            speaker_condition=spk_condition,
            video_text_condition=video_text_condition,
            lip_text_condition=lip_text_condition,
            face_text_condition=face_text_condition,
            video_guidance_scale=video_guidance_scale,
            face_guidance_scale=face_guidance_scale,
            lip_guidance_scale=lip_guidance_scale,
            t_shift=t_shift,
        )

        if prompt_features is None:
            x1_wo_prompt = x1
            x1_prompt = None
            x1_wo_prompt_lens = None

        else:
            x1_wo_prompt_lens = (~padding_mask).sum(-1) - prompt_features_lens
            x1_prompt = torch.zeros(
                x1.size(0), prompt_features_lens.max(), x1.size(2), device=x1.device
            )
            x1_wo_prompt = torch.zeros(
                x1.size(0), x1_wo_prompt_lens.max(), x1.size(2), device=x1.device
            )
            for i in range(x1.size(0)):
                x1_wo_prompt[i, : x1_wo_prompt_lens[i], :] = x1[
                    i,
                    prompt_features_lens[i] : prompt_features_lens[i]
                    + x1_wo_prompt_lens[i],
                ]
                x1_prompt[i, : prompt_features_lens[i], :] = x1[
                    i, : prompt_features_lens[i]
                ]

        return x1_wo_prompt, x1_wo_prompt_lens, x1_prompt, prompt_features_lens
