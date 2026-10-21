"""Multi-head latent attention (MLA) layer."""

from __future__ import annotations

from .layer import MultiHeadLatentAttention

__all__ = ["MultiHeadLatentAttention", "available_backends"]


def available_backends() -> tuple[str, ...]:
    return ("reference",)
