"""Mask helpers shared by the reference implementations.

Conventions used across AttentionHub:

* Boolean masks use ``True`` for positions that are kept.
* Floating point masks are additive (``0`` keep, large negative values drop).
* ``causal=True`` is independent from an explicit mask and can be combined with it.
"""

from __future__ import annotations

import torch


def causal_mask(
    q_len: int,
    kv_len: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a ``(q_len, kv_len)`` boolean causal mask (``True`` = keep)."""
    return torch.ones(q_len, kv_len, device=device, dtype=torch.bool).tril(
        diagonal=kv_len - q_len
    )


def combine_masks(
    mask: torch.Tensor | None,
    causal: bool,
    q_len: int,
    kv_len: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor | None:
    """Combine an optional user mask with an optional causal mask."""
    if not causal:
        return mask

    causal_m = causal_mask(q_len, kv_len, device=device if mask is None else mask.device)
    if mask is None:
        return causal_m
    if mask.dtype == torch.bool:
        return mask & causal_m

    # Additive float mask: masked positions get the dtype minimum.
    neg = torch.finfo(mask.dtype).min
    return mask.masked_fill(~causal_m, neg)


def apply_mask(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply a boolean (``True`` = keep) or additive float mask to scores."""
    if mask.dtype == torch.bool:
        return scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    return scores + mask
