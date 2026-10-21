"""Reference (readable) implementation of scaled dot-product attention."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from ..mask import apply_mask, combine_masks


def scaled_dot_product_attention(
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
    """Compute scaled dot-product attention.

    Args:
        Q: Query tensor of shape ``(..., n_queries, d_k)``.
        K: Key tensor of shape ``(..., n_keys, d_k)``.
        V: Value tensor of shape ``(..., n_keys, d_v)``.
        mask: Optional attention mask. Boolean masks use ``True`` for positions
            that are kept; floating point masks are additive.
        causal: Apply a causal mask on top of ``mask``.
        scale: Softmax scale. Defaults to ``d_k ** -0.5``.
        dropout_p: Dropout probability applied to the attention weights.
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

    d_k = Q.size(-1)
    scale = d_k**-0.5 if scale is None else scale

    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    mask = combine_masks(mask, causal, Q.size(-2), K.size(-2), device=scores.device)
    if mask is not None:
        scores = apply_mask(scores, mask)

    attention_weights = torch.softmax(scores, dim=-1)
    if dropout_p > 0.0:
        attention_weights = torch.nn.functional.dropout(
            attention_weights, p=dropout_p, training=True
        )

    output = torch.matmul(attention_weights, V)
    if return_attn:
        return output, attention_weights
    return output


sdpa = scaled_dot_product_attention
