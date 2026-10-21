"""Predictive sparse attention (PSA) layer."""

from __future__ import annotations

from .layer import PSA, PredictSparseAttention


def available_backends() -> tuple[str, ...]:
    return ("reference",)


__all__ = ["PSA", "PredictSparseAttention", "available_backends"]
