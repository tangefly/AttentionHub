"""Qwen sparse attention (QSA) layer."""

from __future__ import annotations

from .layer import QSAIndexer, QwenSparseAttention


def available_backends() -> tuple[str, ...]:
    return ("reference",)


__all__ = [
    "QSAIndexer",
    "QwenSparseAttention",
    "available_backends",
]
