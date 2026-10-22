"""DeepSeek sparse attention (DSA) layer.

DSA, introduced by DeepSeek-V3.2-Exp, keeps the MLA backbone and adds a small
*indexer* that selects a fixed number of key/value tokens per query.  The main
attention then runs only on the selected subset, so the cost of long-context
attention grows with ``index_topk`` instead of the full context length.

The indexer is deliberately lightweight:

1. project the MLA query latent (``q_resid``) into a few index heads;
2. project the hidden states into a single shared index key and normalize it;
3. apply RoPE to the first ``index_rope_head_dim`` channels;
4. score every key as ``relu(q . k)`` weighted and summed over index heads;
5. keep the ``index_topk`` highest scoring keys per query.

This file is a readable reference layer.  It performs the projections, the
top-k selection and the sparse attention in plain PyTorch.  The layer follows
the same projection and RoPE conventions as MLA, but is implemented
independently and does not inherit from it.  The indexer runs under
``torch.no_grad`` in inference code; production kernels consume the returned
indices directly in a sparse MLA kernel.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from ..mask import apply_mask, causal_mask
from ..norm import RMSNorm
from ..rope import apply_rotary_pos_emb

__all__ = ["DeepseekSparseAttention", "DeepseekSparseIndexer"]


class DeepseekSparseIndexer(nn.Module):
    """Select the ``index_topk`` most relevant keys for every query.

    Args:
        hidden_size: Input hidden dimension.
        query_dim: Dimension of the query latent fed to ``wq_b``.  For a layer
            with query compression this is ``q_lora_rank``; without query
            compression it is ``hidden_size``.
        index_n_heads: Number of indexer heads.
        index_head_dim: Indexer query/key head dimension.
        index_topk: Number of keys kept per query.  Clamped to the KV length.
        index_rope_head_dim: Number of rotary channels.  Defaults to
            ``index_head_dim``; DeepSeek-V3.2 uses the MLA
            ``qk_rope_head_dim`` (64 of 128).  Must be even.
        norm_eps: Epsilon of the key LayerNorm.

    Note:
        Unlike the main DSA attention, the indexer uses the half-split
        (non-interleaved) RoPE layout.  It also normalizes its key with a
        LayerNorm rather than an RMSNorm, matching the released checkpoint.
    """

    def __init__(
        self,
        hidden_size: int,
        query_dim: int,
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        index_rope_head_dim: int | None = None,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if index_rope_head_dim is None:
            index_rope_head_dim = index_head_dim
        if index_n_heads <= 0:
            raise ValueError(
                f"index_n_heads must be positive, got {index_n_heads}"
            )
        if index_head_dim <= 0:
            raise ValueError(
                f"index_head_dim must be positive, got {index_head_dim}"
            )
        if index_topk <= 0:
            raise ValueError(f"index_topk must be positive, got {index_topk}")
        if index_rope_head_dim < 0 or index_rope_head_dim > index_head_dim:
            raise ValueError(
                "index_rope_head_dim must be between 0 and index_head_dim "
                f"({index_head_dim}), got {index_rope_head_dim}"
            )
        if index_rope_head_dim % 2 != 0:
            raise ValueError(
                "index_rope_head_dim must be even for RoPE, "
                f"got {index_rope_head_dim}"
            )

        self.hidden_size = hidden_size
        self.query_dim = query_dim
        self.n_heads = index_n_heads
        self.head_dim = index_head_dim
        self.rope_head_dim = index_rope_head_dim
        self.index_topk = index_topk
        self.softmax_scale = self.head_dim**-0.5

        self.wq_b = nn.Linear(
            query_dim, self.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=norm_eps)
        self.weights_proj = nn.Linear(hidden_size, self.n_heads, bias=False)

    @staticmethod
    def _prepare_mask(
        attention_mask: torch.Tensor | None,
        batch_size: int,
        q_len: int,
        kv_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Normalize a 2D/3D/4D mask to ``(batch, q_len, kv_len)``."""
        if attention_mask is None:
            return causal_mask(q_len, kv_len, device=device)
        mask = attention_mask
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        elif mask.dim() == 4:
            if mask.shape[1] != 1:
                raise ValueError(
                    "a 4D indexer mask must have a singleton head dimension, "
                    f"got shape {tuple(mask.shape)}"
                )
            mask = mask[:, 0]
        if mask.dim() != 3:
            raise ValueError(
                "attention_mask must be broadcastable to "
                f"({batch_size}, {q_len}, {kv_len}), got shape "
                f"{tuple(attention_mask.shape)}"
            )
        try:
            return mask.expand(batch_size, q_len, kv_len)
        except RuntimeError as exc:
            raise ValueError(
                "attention_mask must be broadcastable to "
                f"({batch_size}, {q_len}, {kv_len}), got shape "
                f"{tuple(attention_mask.shape)}"
            ) from exc

    def _prepare_rope(
        self,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            not isinstance(position_embeddings, (tuple, list))
            or len(position_embeddings) != 2
        ):
            raise ValueError("position_embeddings must be a (cos, sin) tuple")
        cos, sin = position_embeddings
        if cos.shape != sin.shape:
            raise ValueError(
                "cos and sin must have the same shape, "
                f"got {cos.shape} and {sin.shape}"
            )
        if cos.dim() == 2:
            cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
        if cos.dim() != 3:
            raise ValueError(
                "cos and sin must have shape (batch, seq_len, rotary_dim) "
                f"or (seq_len, rotary_dim), got {tuple(cos.shape)}"
            )
        if cos.shape[-1] != self.rope_head_dim:
            raise ValueError(
                f"indexer rotary_dim ({cos.shape[-1]}) must equal "
                f"index_rope_head_dim ({self.rope_head_dim})"
            )
        if cos.shape[-2] < seq_length:
            raise ValueError(
                f"position_embeddings cover {cos.shape[-2]} positions, "
                f"but the input has {seq_length} tokens"
            )
        return cos[:, -seq_length:], sin[:, -seq_length:]

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the ``index_topk`` selected key indices.

        Args:
            hidden_states: ``(batch, seq_len, hidden_size)`` input.
            q_resid: Query latent, i.e. ``q_a_layernorm(q_a_proj(x))`` with
                shape ``(batch, seq_len, query_dim)``.
            position_embeddings: ``(cos, sin)`` tables of shape
                ``(batch, seq_len, index_rope_head_dim)``.
            attention_mask: Optional bool (``True`` = keep) or additive float
                mask broadcastable to ``(batch, seq_len, kv_len)``.  A 4D
                ``(batch, 1, seq_len, kv_len)`` mask is also accepted.  When
                omitted, a causal mask is used.

        Returns:
            ``int64`` indices of shape
            ``(batch, seq_len, min(index_topk, kv_len))``.  Indices that fall
            on masked positions are unspecified and should be discarded by the
            consumer; the sparse mask built in
            :class:`DeepseekSparseAttention` always does so.
        """
        batch_size, seq_length, _ = hidden_states.shape
        cos, sin = self._prepare_rope(position_embeddings, seq_length)

        q = self.wq_b(q_resid).view(
            batch_size, seq_length, self.n_heads, self.head_dim
        )
        q_rot, q_pass = torch.split(
            q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )

        k = self.k_norm(self.wk(hidden_states)).unsqueeze(2)
        k_rot, k_pass = torch.split(
            k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )

        # The indexer uses the half-split RoPE layout; ``unsqueeze_dim=2``
        # broadcasts the ``(batch, seq, rope)`` tables over the head dimension.
        q_rot, k_rot = apply_rotary_pos_emb(
            q_rot, k_rot, cos, sin, unsqueeze_dim=2
        )
        q = torch.cat([q_rot, q_pass], dim=-1)
        k = torch.cat([k_rot, k_pass], dim=-1).squeeze(2)

        # Per-head raw scores: (batch, seq, heads, kv_len), non-negative.
        scores = torch.matmul(
            q.float(), k.transpose(-1, -2).float().unsqueeze(1)
        ) * self.softmax_scale
        scores = F.relu(scores)

        # Weight the heads and sum: (batch, seq, 1, heads) @ (...heads, kv)
        weights = self.weights_proj(hidden_states).float() * (
            self.n_heads**-0.5
        )
        index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)

        kv_len = k.shape[1]
        mask = self._prepare_mask(
            attention_mask,
            batch_size,
            seq_length,
            kv_len,
            hidden_states.device,
        )
        if mask.dtype == torch.bool:
            index_scores = index_scores.masked_fill(
                ~mask, torch.finfo(index_scores.dtype).min
            )
        else:
            index_scores = index_scores + mask

        topk = min(self.index_topk, kv_len)
        return index_scores.topk(topk, dim=-1).indices


class DeepseekSparseAttention(nn.Module):
    """DeepSeek sparse attention (DSA) with an MLA-style backbone.

    The layer uses the full MLA parameterization (low-rank query and KV
    compression, decoupled RoPE) and adds a :class:`DeepseekSparseIndexer`.
    The indexer selects ``index_topk`` keys per query; those indices are turned
    into a boolean keep mask and combined with the causal/padding mask before
    the attention softmax.  Attention weights are returned in dense
    ``(batch, heads, seq, kv_len)`` form with zeros at unselected positions.

    This is a standalone implementation.  It deliberately does not inherit from
    :class:`attention.mla.layer.MultiHeadLatentAttention`, so the two
    algorithms can evolve independently while sharing the same parameter names
    and state-dict layout.

    Args:
        hidden_size: Input hidden dimension.
        num_attention_heads: Number of query heads.
        kv_lora_rank: Rank of the compressed KV latent.
        qk_nope_head_dim: Non-rotary part of the query/key head dimension.
        qk_rope_head_dim: Rotary part of the query/key head dimension.  Must be
            even.
        v_head_dim: Value head dimension.
        index_n_heads: Number of indexer heads.
        index_head_dim: Indexer head dimension.
        index_topk: Number of keys retained per query.
        q_lora_rank: Optional rank of the compressed query latent.  When
            ``None``, the indexer consumes the hidden states directly.
        index_rope_head_dim: Number of rotary channels used by the indexer.
            Defaults to ``qk_rope_head_dim``.  It must be even and no larger
            than ``index_head_dim``.
        attention_dropout: Dropout probability on the attention weights.
        attention_bias: Whether the input projections and ``o_proj`` use bias.
        rms_norm_eps: Epsilon for the query and KV latent RMSNorms.
        scaling: Optional main-attention scale.  Defaults to
            ``qk_head_dim ** -0.5``.
        rope_layout: RoPE layout of the main attention, ``"interleaved"``
            (DeepSeek checkpoints, default) or ``"half"``.  The indexer always
            uses the half-split layout.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        q_lora_rank: int | None = None,
        index_rope_head_dim: int | None = None,
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
            raise ValueError(
                f"q_lora_rank must be positive or None, got {q_lora_rank}"
            )
        if rope_layout not in ("interleaved", "half"):
            raise ValueError(
                "rope_layout must be 'interleaved' or 'half', "
                f"got {rope_layout!r}"
            )
        if index_rope_head_dim is None:
            index_rope_head_dim = qk_rope_head_dim

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
        self.index_topk = index_topk

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
        # the compressed representation is always one; it is expanded after the
        # KV up-projection.
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

        self.indexer = DeepseekSparseIndexer(
            hidden_size=hidden_size,
            query_dim=q_lora_rank if q_lora_rank is not None else hidden_size,
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            index_topk=index_topk,
            index_rope_head_dim=index_rope_head_dim,
            norm_eps=rms_norm_eps,
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
            and ``value_states`` of shape ``(batch, num_heads, seq_len, v_head_dim)``.
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
        # Allow a cache that covers more positions than the current input and
        # use the last ``seq_length`` entries, as done by the MLA layer.
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
        """Run DSA.

        Args:
            hidden_states: ``(batch, seq_len, hidden_size)`` input.
            position_embeddings: ``(cos, sin)`` tables of shape
                ``(batch, seq_len, qk_rope_head_dim)``; see
                :func:`attention.rope.build_rope_cache_batched`.
            attention_mask: Optional bool mask (``True`` = keep) or additive
                float mask, broadcastable to
                ``(batch, 1, seq_len, seq_len)``.  When omitted, a causal mask
                is used.

        Returns:
            ``(attn_output, attn_weights)`` with shapes
            ``(batch, seq_len, hidden_size)`` and
            ``(batch, num_heads, seq_len, seq_len)``.
        """
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)

        if self.q_lora_rank is None:
            q_resid = hidden_states
            q = self.q_proj(hidden_states)
        else:
            q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
            q = self.q_b_proj(q_resid)
        q = q.view(query_shape).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_nope, k_pe = torch.split(
            compressed_kv,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        kv_nope = self.kv_a_layernorm(kv_nope).view(
            batch_size, 1, seq_length, self.kv_lora_rank
        )
        k_pe = k_pe.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        cos, sin = self.prepare_position_embeddings(
            position_embeddings, seq_length
        )
        q_pe, k_pe = self.apply_rope(q_pe, k_pe, cos, sin)

        query_states = torch.cat((q_nope, q_pe), dim=-1)
        key_states, value_states = self.expand_kv(kv_nope, k_pe)
        kv_len = key_states.shape[2]

        if attention_mask is None:
            attention_mask = causal_mask(
                seq_length, kv_len, device=hidden_states.device
            )
        if attention_mask.dim() == 4 and attention_mask.shape[1] != 1:
            raise ValueError(
                "DSA needs a per-query mask shared by all heads; a 4D "
                "attention_mask must have shape (batch, 1, seq_len, kv_len), "
                f"got {tuple(attention_mask.shape)}"
            )
        index_mask = (
            attention_mask[:, 0]
            if attention_mask.dim() == 4
            else attention_mask
        )
        topk_indices = self.indexer(
            hidden_states,
            q_resid,
            (cos, sin),
            index_mask,
        )

        # ``keep`` is a boolean mask that is True at the selected keys.
        keep = torch.zeros(
            batch_size,
            1,
            seq_length,
            kv_len,
            dtype=torch.bool,
            device=hidden_states.device,
        ).scatter(-1, topk_indices.unsqueeze(1), True)
        if attention_mask.dtype == torch.bool:
            sparse_mask = attention_mask & keep
        else:
            sparse_mask = attention_mask.masked_fill(
                ~keep, torch.finfo(attention_mask.dtype).min
            )

        attn_output, attn_weights = self.eager_attention(
            query_states,
            key_states,
            value_states,
            sparse_mask,
        )
        attn_output = attn_output.reshape(
            batch_size, seq_length, -1
        ).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
