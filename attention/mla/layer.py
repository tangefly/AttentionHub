"""Multi-head latent attention (MLA) layer.

MLA, introduced by DeepSeek-V2, compresses the key/value states into a shared
low-rank latent and keeps a small decoupled RoPE key.  The compressed latent is
cached instead of per-head key/value tensors, which is what makes the KV cache
of DeepSeek models small.

This module is a self-contained reference layer: it performs the projections,
the latent normalization and the attention computation in plain PyTorch, and it
reuses the hub's shared :mod:`attention.rope`, :mod:`attention.norm` and
:mod:`attention.mask` utilities.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from ..mask import apply_mask, causal_mask
from ..norm import RMSNorm
from ..rope import apply_rotary_pos_emb

__all__ = ["MultiHeadLatentAttention"]


class MultiHeadLatentAttention(nn.Module):
    """Multi-head latent attention as used by DeepSeek-V2.

    Args:
        hidden_size: Input hidden dimension.
        num_attention_heads: Number of query heads.
        kv_lora_rank: Rank of the compressed KV latent.
        qk_nope_head_dim: Non-rotary part of the query/key head dimension.
        qk_rope_head_dim: Rotary part of the query/key head dimension. Must be
            even because it is rotated by RoPE.
        v_head_dim: Value head dimension.
        q_lora_rank: Optional rank of the compressed query latent. When
            ``None``, a direct query projection is used.
        attention_dropout: Dropout probability on the attention weights.
        attention_bias: Whether the input projections and ``o_proj`` use bias.
        rms_norm_eps: Epsilon for the query and KV latent RMSNorms.
        scaling: Optional attention scale. Defaults to
            ``(qk_nope_head_dim + qk_rope_head_dim) ** -0.5``; models using
            YaRN can pass the already-scaled value.
        rope_layout: RoPE channel layout.  ``"interleaved"`` (default) matches
            the complex RoPE used by DeepSeek-V2 checkpoints: adjacent
            ``(2i, 2i + 1)`` channels form one rotation pair.  ``"half"``
            matches the hub/Llama convention used by
            :func:`attention.rope.apply_rotary_pos_emb`, where the first and
            second halves of the rotary dimension are paired.

    Note:
        Both layouts consume the cos/sin tables produced by
        :func:`attention.rope.build_rope_cache_batched`.  The interleaved
        layout is implemented with the shared RoPE function plus a channel
        permutation, so no separate RoPE kernel is needed.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None = None,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        rms_norm_eps: float = 1e-6,
        scaling: float | None = None,
        rope_layout: Literal["interleaved", "half"] = "interleaved",
    ) -> None:
        super().__init__()
        if qk_rope_head_dim % 2 != 0:
            raise ValueError(
                f"qk_rope_head_dim must be even for RoPE, got {qk_rope_head_dim}"
            )
        if q_lora_rank is not None and q_lora_rank <= 0:
            raise ValueError(f"q_lora_rank must be positive or None, got {q_lora_rank}")
        if rope_layout not in ("interleaved", "half"):
            raise ValueError(
                "rope_layout must be 'interleaved' or 'half', "
                f"got {rope_layout!r}"
            )

        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.attention_dropout = attention_dropout
        self.rope_layout = rope_layout
        self.scaling = (
            scaling if scaling is not None else self.qk_head_dim ** (-0.5)
        )
        self.is_causal = True

        self.q_proj = (
            None
            if q_lora_rank is not None
            else nn.Linear(hidden_size, self.num_heads * self.qk_head_dim, bias=False)
        )
        self.q_a_proj = (
            nn.Linear(hidden_size, q_lora_rank, bias=attention_bias)
            if q_lora_rank is not None
            else None
        )
        self.q_a_layernorm = (
            RMSNorm(q_lora_rank, eps=rms_norm_eps) if q_lora_rank is not None else None
        )
        self.q_b_proj = (
            nn.Linear(q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)
            if q_lora_rank is not None
            else None
        )

        # The KV latent is shared by every head, so the "KV heads" dimension of
        # MLA is always one; it is expanded after the KV up-projection.
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden_size,
            kv_lora_rank + qk_rope_head_dim,
            bias=attention_bias,
        )
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            kv_lora_rank,
            self.num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * v_head_dim,
            hidden_size,
            bias=attention_bias,
        )

    def expand_kv(
        self,
        kv_nope: torch.Tensor,
        k_rot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand the compressed KV latent into per-head key/value states.

        Args:
            kv_nope: Normalized latent without positional information, shape
                ``(batch, 1, seq_len, kv_lora_rank)``.
            k_rot: Shared RoPE key, shape
                ``(batch, 1, seq_len, qk_rope_head_dim)``.

        Returns:
            ``key_states`` of shape
            ``(batch, num_heads, seq_len, qk_nope_head_dim + qk_rope_head_dim)``
            and ``value_states`` of shape
            ``(batch, num_heads, seq_len, v_head_dim)``.
        """
        batch_size, _, seq_length, _ = kv_nope.shape
        key_shape = (
            batch_size,
            seq_length,
            -1,
            self.qk_nope_head_dim + self.v_head_dim,
        )

        kv_nope = self.kv_b_proj(kv_nope).view(key_shape).transpose(1, 2)
        k_nope, value_states = torch.split(
            kv_nope,
            [self.qk_nope_head_dim, self.v_head_dim],
            dim=-1,
        )
        # This only changes the view: all heads share the same RoPE key.
        k_rot = k_rot.expand(-1, k_nope.shape[1], -1, -1)

        key_states = kv_nope.new_empty(
            *kv_nope.shape[:-1],
            self.qk_nope_head_dim + self.qk_rope_head_dim,
        )
        key_states[..., : self.qk_nope_head_dim].copy_(k_nope)
        key_states[..., self.qk_nope_head_dim :].copy_(k_rot)
        return key_states, value_states

    @staticmethod
    def to_half_layout(x: torch.Tensor) -> torch.Tensor:
        """Reorder interleaved pairs to the ``[even, odd]`` half layout."""
        return torch.cat((x[..., 0::2], x[..., 1::2]), dim=-1)

    @staticmethod
    def from_half_layout(x: torch.Tensor) -> torch.Tensor:
        """Reorder a ``[even, odd]`` half layout back to interleaved pairs."""
        half = x.shape[-1] // 2
        return torch.stack((x[..., :half], x[..., half:]), dim=-1).flatten(-2)

    def apply_rope(
        self,
        q_pe: torch.Tensor,
        k_pe: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rope_layout == "half":
            return apply_rotary_pos_emb(q_pe, k_pe, cos, sin)

        # DeepSeek rotates adjacent channel pairs, while the shared RoPE
        # implementation rotates the two halves of the vector.  Permuting the
        # channels before and after the call maps one convention onto the other.
        q_pe = self.to_half_layout(q_pe)
        k_pe = self.to_half_layout(k_pe)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin)
        return self.from_half_layout(q_pe), self.from_half_layout(k_pe)

    def prepare_position_embeddings(
        self,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(position_embeddings, (tuple, list)) or len(position_embeddings) != 2:
            raise ValueError(
                "position_embeddings must be a (cos, sin) tuple"
            )
        cos, sin = position_embeddings
        if cos.shape != sin.shape:
            raise ValueError(
                f"cos and sin must have the same shape, got {cos.shape} and {sin.shape}"
            )
        if cos.dim() == 2:
            cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
        if cos.dim() != 3:
            raise ValueError(
                "cos and sin must have shape (batch, seq_len, rotary_dim) "
                f"or (seq_len, rotary_dim), got {tuple(cos.shape)}"
            )
        if cos.shape[-1] != self.qk_rope_head_dim:
            raise ValueError(
                f"rotary_dim ({cos.shape[-1]}) must equal "
                f"qk_rope_head_dim ({self.qk_rope_head_dim})"
            )
        if cos.shape[-2] < seq_length:
            raise ValueError(
                f"position_embeddings cover {cos.shape[-2]} positions, "
                f"but the input has {seq_length} tokens"
            )
        # Allow a cache that covers more positions than the current input, as
        # done by the QSA layer, and use the last ``seq_length`` entries.
        return cos[:, -seq_length:], sin[:, -seq_length:]

    def eager_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) * self.scaling
        attn_weights = apply_mask(attn_weights, attention_mask)

        attn_weights = torch.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights,
            p=0.0 if not self.training else self.attention_dropout,
            training=self.training,
        )
        attn_output = torch.matmul(attn_weights, value_states)
        return attn_output.transpose(1, 2).contiguous(), attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run MLA.

        Args:
            hidden_states: ``(batch, seq_len, hidden_size)`` input.
            position_embeddings: ``(cos, sin)`` tables of shape
                ``(batch, seq_len, qk_rope_head_dim)``; see
                :func:`attention.rope.build_rope_cache_batched`.
            attention_mask: Optional boolean mask (``True`` = keep) or additive
                float mask, broadcastable to
                ``(batch, num_heads, seq_len, seq_len)``.  When omitted, a
                causal mask is used.

        Returns:
            ``(attn_output, attn_weights)`` with shapes
            ``(batch, seq_len, hidden_size)`` and
            ``(batch, num_heads, seq_len, seq_len)``.
        """
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(
                self.q_a_layernorm(self.q_a_proj(hidden_states))
            )
        q = q.view(query_shape).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_nope, k_pe = torch.split(
            compressed_kv,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        # Both latents are single-head, 4D tensors so that every head shares
        # the same compressed representation.
        k_nope = self.kv_a_layernorm(kv_nope).view(
            batch_size, 1, seq_length, self.kv_lora_rank
        )
        k_pe = k_pe.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        cos, sin = self.prepare_position_embeddings(
            position_embeddings, seq_length
        )
        q_pe, k_pe = self.apply_rope(q_pe, k_pe, cos, sin)

        query_states = torch.cat((q_nope, q_pe), dim=-1)
        key_states, value_states = self.expand_kv(k_nope, k_pe)

        if attention_mask is None:
            attention_mask = causal_mask(
                seq_length,
                seq_length,
                device=hidden_states.device,
            )

        attn_output, attn_weights = self.eager_attention(
            query_states,
            key_states,
            value_states,
            attention_mask,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
