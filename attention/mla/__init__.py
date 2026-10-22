"""Multi-head latent attention (MLA) layer."""

from __future__ import annotations

from .layer import MultiHeadLatentAttention

# Backwards-compatible name used by the README and the lazy package aliases.
DeepseekV2Attention = MultiHeadLatentAttention

__all__ = [
    "DeepseekV2Attention",
    "MultiHeadLatentAttention",
    "available_backends",
]


def available_backends() -> tuple[str, ...]:
    return ("reference",)
