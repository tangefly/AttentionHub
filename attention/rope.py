"""Rotary position embedding (RoPE) utilities.

Two cache layouts are provided:

* :func:`build_rope_cache` returns ``(seq_len, dim)`` tables.
* :func:`build_rope_cache_batched` returns ``(batch, seq_len, dim)`` tables,
  which can be passed directly to :func:`apply_rotary_pos_emb` for 4D
  ``(batch, heads, seq_len, head_dim)`` query/key tensors.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

__all__ = [
    "apply_rotary_pos_emb",
    "build_rope_cache",
    "build_rope_cache_batched",
    "rotate_half",
]


def rotate_half(x: Tensor) -> Tensor:
    """Rotate half of the last dimension: ``[x1, x2] -> [-x2, x1]``."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def build_rope_cache(
    seq_len: int,
    dim: int,
    *,
    base: float = 10000.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[Tensor, Tensor]:
    """Build RoPE ``(cos, sin)`` tables of shape ``(seq_len, dim)``.

    Args:
        seq_len: Number of positions.
        dim: Rotary dimension. Must be even.
        base: RoPE theta base (default ``10000.0``).
        device: Device for the returned tensors.
        dtype: Dtype for the returned tensors. Frequencies are always computed
            in float32 and cast at the end.
    """
    if dim <= 0:
        raise ValueError(f"RoPE dim must be positive, got {dim}")
    if dim % 2 != 0:
        raise ValueError(f"RoPE dim must be even, got {dim}")
    if seq_len < 0:
        raise ValueError(f"seq_len must be non-negative, got {seq_len}")
    if base <= 0:
        raise ValueError(f"base must be positive, got {base}")

    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, device=device, dtype=torch.float32)
            / dim
        )
    )
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def build_rope_cache_batched(
    batch_size: int,
    seq_len: int,
    dim: int,
    *,
    base: float = 10000.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[Tensor, Tensor]:
    """Build RoPE ``(cos, sin)`` tables of shape ``(batch, seq_len, dim)``."""
    if batch_size < 0:
        raise ValueError(f"batch_size must be non-negative, got {batch_size}")
    cos, sin = build_rope_cache(
        seq_len, dim, base=base, device=device, dtype=dtype
    )
    return (
        cos.unsqueeze(0).expand(batch_size, -1, -1),
        sin.unsqueeze(0).expand(batch_size, -1, -1),
    )


def apply_rotary_pos_emb(
    q: Tensor,
    k: Optional[Tensor] = None,
    cos: Optional[Tensor] = None,
    sin: Optional[Tensor] = None,
    unsqueeze_dim: int = 1,
) -> Tensor | Tuple[Tensor, Tensor]:
    """Apply RoPE to ``q`` (and optionally ``k``).

    Args:
        q: Query tensor, typically ``(batch, heads, seq_len, head_dim)``.
        k: Optional key tensor with the same shape as ``q``.
        cos: Cosine table of shape ``(batch, seq_len, rotary_dim)`` or
            ``(1, seq_len, rotary_dim)``.
        sin: Sine table with the same shape as ``cos``.
        unsqueeze_dim: Dimension used to broadcast ``cos`` / ``sin``. The
            default ``1`` matches ``(batch, heads, seq_len, head_dim)``.

    Returns:
        Rotated ``q``, or ``(q, k)`` when ``k`` is given.

    The part of the head dimension beyond ``cos.shape[-1]`` is left unchanged
    (partial RoPE).
    """
    if cos is None or sin is None:
        raise ValueError("cos and sin are required")
    if cos.shape != sin.shape:
        raise ValueError(
            f"cos and sin must have the same shape, got {cos.shape} and {sin.shape}"
        )
    if q.shape[-1] < cos.shape[-1]:
        raise ValueError(
            f"q head_dim ({q.shape[-1]}) is smaller than rotary_dim ({cos.shape[-1]})"
        )

    cos = cos.to(device=q.device, dtype=q.dtype).unsqueeze(unsqueeze_dim)
    sin = sin.to(device=q.device, dtype=q.dtype).unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]

    q_rope, q_nope = q[..., :rotary_dim], q[..., rotary_dim:]
    q_rope = (q_rope * cos) + (rotate_half(q_rope) * sin)
    q_rotated = torch.cat([q_rope, q_nope], dim=-1)

    if k is None:
        return q_rotated

    if k.shape[-1] < rotary_dim:
        raise ValueError(
            f"k head_dim ({k.shape[-1]}) is smaller than rotary_dim ({rotary_dim})"
        )
    k_rope, k_nope = k[..., :rotary_dim], k[..., rotary_dim:]
    k_rope = (k_rope * cos) + (rotate_half(k_rope) * sin)
    k_rotated = torch.cat([k_rope, k_nope], dim=-1)
    return q_rotated, k_rotated
