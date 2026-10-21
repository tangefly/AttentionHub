"""Factorized attention."""

from __future__ import annotations

from torch import Tensor

from ..dispatch import resolve_backend
from .reference import factorized_attention as _factorized_attention


def available_backends() -> tuple[str, ...]:
    return ("reference",)


def fta(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    l: int,
    *,
    backend: str = "auto",
    return_attn: bool = False,
) -> Tensor:
    """Compute factorized attention.

    Args:
        Q, K, V: Tensors of shape ``(batch, head, seq, dim)``.
        l: Local window size / stride.
        backend: Currently only ``"reference"`` is implemented.
        return_attn: Not supported; FTA does not expose a single attention
            matrix.
    """
    resolve_backend(backend, available_backends(), "fta")
    if return_attn:
        raise ValueError(
            "the fta reference does not expose a single attention matrix; "
            "return_attn is not supported"
        )
    return _factorized_attention(Q, K, V, l)


factorized_attention = fta

__all__ = ["available_backends", "factorized_attention", "fta"]
