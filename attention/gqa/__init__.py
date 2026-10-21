"""Grouped-query attention."""

from __future__ import annotations

from typing import Optional

from torch import Tensor

from ..dispatch import resolve_backend
from .reference import grouped_query_attention as _grouped_query_attention


def available_backends() -> tuple[str, ...]:
    return ("reference",)


def gqa(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Optional[Tensor] = None,
    *,
    causal: bool = False,
    scale: Optional[float] = None,
    dropout_p: float = 0.0,
    backend: str = "auto",
    return_attn: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Compute grouped-query attention.

    See :func:`attention.gqa.reference.grouped_query_attention` for the tensor
    layout and mask conventions.
    """
    resolve_backend(backend, available_backends(), "gqa")
    return _grouped_query_attention(
        Q,
        K,
        V,
        mask=mask,
        causal=causal,
        scale=scale,
        dropout_p=dropout_p,
        return_attn=return_attn,
    )


grouped_query_attention = gqa

__all__ = ["available_backends", "gqa", "grouped_query_attention"]
