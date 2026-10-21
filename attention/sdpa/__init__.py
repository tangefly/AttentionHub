"""Scaled dot-product attention with selectable backends."""

from __future__ import annotations

from typing import Optional

from torch import Tensor

from ..dispatch import resolve_backend
from . import reference, torch as torch_backend

_BACKENDS = {
    "reference": reference.sdpa,
    "torch": torch_backend.sdpa,
}


def available_backends() -> tuple[str, ...]:
    return ("reference", "torch")


def sdpa(
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
    """Compute scaled dot-product attention.

    Args:
        Q, K, V: Query, key and value tensors.
        mask: Boolean (``True`` = keep) or additive float mask.
        causal: Apply a causal mask on top of ``mask``.
        scale: Softmax scale, defaults to ``d_k ** -0.5``.
        dropout_p: Dropout probability on attention weights.
        backend: ``"auto"``, ``"reference"`` or ``"torch"``.
        return_attn: Return attention weights as well (reference backend only).

    Returns:
        The attention output, or ``(output, attention_weights)`` when
        ``return_attn=True``.
    """
    name = resolve_backend(backend, available_backends(), "sdpa")
    return _BACKENDS[name](
        Q,
        K,
        V,
        mask=mask,
        causal=causal,
        scale=scale,
        dropout_p=dropout_p,
        return_attn=return_attn,
    )


scaled_dot_product_attention = sdpa

__all__ = ["available_backends", "scaled_dot_product_attention", "sdpa"]
