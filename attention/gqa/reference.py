"""Reference implementation of grouped-query attention."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from ..mask import apply_mask, combine_masks


def grouped_query_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Optional[Tensor] = None,
    *,
    causal: bool = False,
    scale: Optional[float] = None,
    dropout_p: float = 0.0,
    return_attn: bool = True,
) -> Tensor | Tuple[Tensor, Tensor]:
    """Compute grouped-query attention.

    Args:
        Q: Query tensor of shape ``(batch, q_heads, q_len, qk_dim)``.
        K: Key tensor of shape ``(batch, kv_heads, kv_len, qk_dim)``.
        V: Value tensor of shape ``(batch, kv_heads, kv_len, v_dim)``.
        mask: Boolean (``True`` = keep) or additive float mask.
        causal: Apply a causal mask on top of ``mask``.
        scale: Softmax scale, defaults to ``qk_dim ** -0.5``.
        dropout_p: Dropout probability on attention weights.
        return_attn: Return ``(output, attention_weights)`` when ``True``.

    Returns:
        The attention output, or ``(output, attention_weights)``.
    """
    if Q.dim() != 4 or K.dim() != 4 or V.dim() != 4:
        raise ValueError(
            "GQA expects Q=(batch, q_heads, q_len, qk_dim), "
            "K=(batch, kv_heads, kv_len, qk_dim) and "
            "V=(batch, kv_heads, kv_len, v_dim)"
        )
    if Q.size(-1) != K.size(-1):
        raise ValueError(
            f"Q and K must have the same feature dim, got {Q.size(-1)} and {K.size(-1)}"
        )
    if K.size(-2) != V.size(-2):
        raise ValueError(
            f"K and V must have the same sequence length, got {K.size(-2)} and {V.size(-2)}"
        )

    batch, q_heads, q_len, qk_dim = Q.shape
    kv_heads, kv_len = K.size(1), K.size(2)
    if q_heads % kv_heads != 0:
        raise ValueError(f"q_heads ({q_heads}) must be divisible by kv_heads ({kv_heads})")
    if Q.size(0) != K.size(0) or K.size(0) != V.size(0):
        raise ValueError("Q, K and V must have the same batch size")

    group_size = q_heads // kv_heads
    Q = Q.view(batch, kv_heads, group_size, q_len, qk_dim)
    scale = qk_dim**-0.5 if scale is None else scale

    scores = torch.einsum("bhgnd,bhmd->bhgnm", Q, K) * scale
    mask = combine_masks(mask, causal, q_len, kv_len, device=scores.device)
    if mask is not None:
        scores = apply_mask(scores, mask)

    attention_weights = torch.softmax(scores, dim=-1)
    if dropout_p > 0.0:
        attention_weights = torch.nn.functional.dropout(
            attention_weights, p=dropout_p, training=True
        )

    output = torch.einsum("bhgnm,bhmd->bhgnd", attention_weights, V)
    output = output.reshape(batch, q_heads, q_len, V.size(-1))
    attention_weights = attention_weights.reshape(batch, q_heads, q_len, kv_len)

    if return_attn:
        return output, attention_weights
    return output


gqa = grouped_query_attention
