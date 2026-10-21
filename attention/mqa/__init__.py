"""Multi-query attention."""

from __future__ import annotations

from typing import Optional

from torch import Tensor

from ..dispatch import resolve_backend
from .reference import multi_query_attention as _multi_query_attention


def available_backends() -> tuple[str, ...]:
    return ("reference",)


def mqa(
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
    """Compute multi-query attention.

    See :func:`attention.mqa.reference.multi_query_attention` for the tensor
    layout and mask conventions.
    """
    resolve_backend(backend, available_backends(), "mqa")
    return _multi_query_attention(
        Q,
        K,
        V,
        mask=mask,
        causal=causal,
        scale=scale,
        dropout_p=dropout_p,
        return_attn=return_attn,
    )


multi_query_attention = mqa

__all__ = ["available_backends", "mqa", "multi_query_attention"]
