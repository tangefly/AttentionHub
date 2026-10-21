"""PyTorch SDPA backend.

This backend uses :func:`torch.nn.functional.scaled_dot_product_attention`,
which dispatches to PyTorch's fused CPU/CUDA kernels when available.
"""

from __future__ import annotations

from typing import Optional

import torch.nn.functional as F
from torch import Tensor

from ..mask import combine_masks


def sdpa(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Optional[Tensor] = None,
    *,
    causal: bool = False,
    scale: Optional[float] = None,
    dropout_p: float = 0.0,
    return_attn: bool = False,
) -> Tensor:
    """Compute SDPA with PyTorch's built-in kernel.

    ``return_attn=True`` is not supported because the fused kernel does not
    materialize the attention matrix.  Use ``backend="reference"`` when weights
    are needed.
    """
    if return_attn:
        raise ValueError(
            "the torch backend does not materialize attention weights; "
            "use backend='reference'"
        )

    if mask is not None and causal:
        mask = combine_masks(mask, True, Q.size(-2), K.size(-2), device=Q.device)
        causal = False

    return F.scaled_dot_product_attention(
        Q,
        K,
        V,
        attn_mask=mask,
        dropout_p=dropout_p,
        is_causal=causal,
        scale=scale,
    )
