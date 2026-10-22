"""DeepSeek sparse attention (DSA) layer."""

from __future__ import annotations

from .layer import DeepseekSparseAttention, DeepseekSparseIndexer

# Backwards-compatible / descriptive alias.
DeepseekV32Attention = DeepseekSparseAttention

__all__ = [
    "DeepseekSparseAttention",
    "DeepseekSparseIndexer",
    "DeepseekV32Attention",
    "available_backends",
]


def available_backends() -> tuple[str, ...]:
    return ("reference",)
