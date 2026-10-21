"""Mixture of Block Attention (MoBA) for variable-length sequences."""

from __future__ import annotations

import torch

from ..dispatch import module_available, resolve_backend
from .reference import moba as _reference_moba


def _flash_available() -> bool:
    return torch.cuda.is_available() and module_available("flash_attn")


def available_backends() -> tuple[str, ...]:
    backends = ["reference"]
    if _flash_available():
        backends.append("flash")
    return tuple(backends)


def moba(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    moba_chunk_size: int,
    moba_topk: int,
    *,
    backend: str = "auto",
) -> torch.Tensor:
    """Compute MoBA attention for packed variable-length sequences.

    Args:
        q, k, v: Tensors of shape ``(total_tokens, heads, head_dim)``.
        cu_seqlens: Cumulative sequence lengths, same convention as flash-attn.
        max_seqlen: Maximum sequence length in the batch.
        moba_chunk_size: KV chunk size.
        moba_topk: Number of chunks each query attends to, including its own.
        backend: ``"auto"``, ``"reference"`` or ``"flash"``.

    Returns:
        Output tensor of shape ``(total_tokens, heads, head_dim)``.
    """
    name = resolve_backend(backend, available_backends(), "moba")
    if name == "flash":
        from .flash import moba as _flash_moba

        return _flash_moba(
            q, k, v, cu_seqlens, max_seqlen, moba_chunk_size, moba_topk
        )
    return _reference_moba(
        q, k, v, cu_seqlens, max_seqlen, moba_chunk_size, moba_topk
    )


moba_attn_varlen = moba
moba_attn_varlen_naive = _reference_moba

__all__ = [
    "available_backends",
    "moba",
    "moba_attn_varlen",
    "moba_attn_varlen_naive",
]
