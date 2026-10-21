"""Qwen sparse attention (QSA) layer."""

from __future__ import annotations

from .layer import (
    QSA,
    QSALayer,
    QSAIndexer,
    Qwen4ExpTextAttention,
    QwenSparseAttention,
)


def available_backends() -> tuple[str, ...]:
    return ("reference",)


__all__ = [
    "QSA",
    "QSALayer",
    "QSAIndexer",
    "Qwen4ExpTextAttention",
    "QwenSparseAttention",
    "available_backends",
]
