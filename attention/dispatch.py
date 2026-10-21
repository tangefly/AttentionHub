"""Backend selection helpers."""

from __future__ import annotations

import importlib.util

BACKEND_ORDER = ("cuda", "triton", "flash", "torch", "reference")


def module_available(name: str) -> bool:
    """Return whether an optional top-level module can be imported."""
    return importlib.util.find_spec(name) is not None


def resolve_backend(
    backend: str,
    available: tuple[str, ...],
    algorithm: str,
) -> str:
    """Resolve ``backend="auto"`` against the backends an algorithm supports."""
    if backend == "auto":
        for candidate in BACKEND_ORDER:
            if candidate in available:
                return candidate
        raise RuntimeError(f"no backend available for {algorithm!r}")

    if backend not in available:
        raise ValueError(
            f"backend {backend!r} is not available for {algorithm!r}; "
            f"choose from {available}"
        )
    return backend
