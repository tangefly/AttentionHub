"""Reference implementation of multi-query attention."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from ..mask import apply_mask, combine_masks


def multi_query_attention(
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
    """Compute multi-query attention (all query heads share one KV head).

    Args:
        Q: Query tensor of shape ``(batch, heads, n_queries, d_k)`` or
            ``(heads, n_queries, d_k)``.
        K: Key tensor of shape ``(batch, n_keys, d_k)`` or ``(n_keys, d_k)``.
        V: Value tensor of shape ``(batch, n_keys, d_v)`` or ``(n_keys, d_v)``.
        mask: Boolean (``True`` = keep) or additive float mask.
        causal: Apply a causal mask on top of ``mask``.
        scale: Softmax scale, defaults to ``d_k ** -0.5``.
        dropout_p: Dropout probability on attention weights.
        return_attn: Return ``(output, attention_weights)`` when ``True``.

    Returns:
        The attention output, or ``(output, attention_weights)``.
    """
    if Q.size(-1) != K.size(-1):
        raise ValueError(
            f"Q and K must have the same feature dim, got {Q.size(-1)} and {K.size(-1)}"
        )
    if K.size(-2) != V.size(-2):
        raise ValueError(
            f"K and V must have the same sequence length, got {K.size(-2)} and {V.size(-2)}"
        )

    squeeze_q = Q.dim() == 3
    squeeze_kv = K.dim() == 2
    if squeeze_q:
        Q = Q.unsqueeze(0)
    if squeeze_kv:
        K = K.unsqueeze(0)
        V = V.unsqueeze(0)

    if Q.dim() != 4 or K.dim() != 3:
        raise ValueError(
            "MQA expects Q=(batch, heads, n_queries, d_k) and "
            "K/V=(batch, n_keys, d_k)"
        )
    if Q.size(0) != K.size(0):
        raise ValueError("Q and K must have the same batch size")

    d_k = Q.size(-1)
    scale = d_k**-0.5 if scale is None else scale

    scores = torch.einsum("bhid,bjd->bhij", Q, K) * scale
    mask = combine_masks(mask, causal, Q.size(-2), K.size(-2), device=scores.device)
    if mask is not None:
        scores = apply_mask(scores, mask)

    attention_weights = torch.softmax(scores, dim=-1)
    if dropout_p > 0.0:
        attention_weights = torch.nn.functional.dropout(
            attention_weights, p=dropout_p, training=True
        )

    output = torch.einsum("bhij,bjd->bhid", attention_weights, V)
    if squeeze_q:
        output = output.squeeze(0)
        attention_weights = attention_weights.squeeze(0)

    if return_attn:
        return output, attention_weights
    return output


mqa = multi_query_attention
