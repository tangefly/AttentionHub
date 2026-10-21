"""AttentionHub: reference and fast implementations of attention variants.

The package uses lazy imports so that optional backends (``flash-attn``,
``triton``, custom CUDA extensions) are only required when they are actually
used.  Importing :mod:`attention` must never fail because one backend is
missing.
"""

from __future__ import annotations

import importlib

__version__ = "0.1.0"

__all__ = [
    "KERNELS",
    "KernelSpec",
    "__version__",
    "available_backends",
    "get_spec",
    "list_kernels",
    "markdown_table",
    # lazy modules
    "sdpa",
    "mqa",
    "gqa",
    "fta",
    "psa",
    "moba",
    "qsa",
    # compatibility aliases
    "scaled_dot_product_attention",
    "multi_query_attention",
    "grouped_query_attention",
    "factorized_attention",
    "PredictSparseAttention",
    "PSA",
    "moba_attn_varlen_naive",
    "moba_attn_varlen",
    "QwenSparseAttention",
    "QSAIndexer",
]

_LAZY_MODULES = {
    "sdpa": ".sdpa",
    "mqa": ".mqa",
    "gqa": ".gqa",
    "fta": ".fta",
    "psa": ".psa",
    "moba": ".moba",
    "qsa": ".qsa",
}

# name -> (relative module, attribute)
_LAZY_ATTRS = {
    "KERNELS": (".registry", "KERNELS"),
    "KernelSpec": (".registry", "KernelSpec"),
    "get_spec": (".registry", "get_spec"),
    "markdown_table": (".registry", "markdown_table"),
    "scaled_dot_product_attention": (".sdpa", "sdpa"),
    "multi_query_attention": (".mqa", "mqa"),
    "grouped_query_attention": (".gqa", "gqa"),
    "factorized_attention": (".fta", "fta"),
    "PredictSparseAttention": (".psa", "PredictSparseAttention"),
    "PSA": (".psa", "PSA"),
    "moba_attn_varlen_naive": (".moba.reference", "moba"),
    "moba_attn_varlen": (".moba", "moba_attn_varlen"),
    "QwenSparseAttention": (".qsa", "QwenSparseAttention"),
    "QSAIndexer": (".qsa", "QSAIndexer"),
}


def __getattr__(name: str):
    if name in _LAZY_MODULES:
        module = importlib.import_module(_LAZY_MODULES[name], __name__)
        globals()[name] = module
        return module
    if name in _LAZY_ATTRS:
        module_name, attribute = _LAZY_ATTRS[name]
        module = importlib.import_module(module_name, __name__)
        value = getattr(module, attribute)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_MODULES) | set(_LAZY_ATTRS))


def list_kernels() -> tuple[str, ...]:
    """Return the canonical short names of all registered kernels."""
    from .registry import KERNELS

    return tuple(sorted(KERNELS))


def available_backends(algorithm: str) -> tuple[str, ...]:
    """Return the installed/usable backends for ``algorithm``.

    This performs a lazy import of the algorithm package but never imports an
    optional backend implementation unless its availability check requires it.
    """
    from .registry import KERNELS

    if algorithm not in KERNELS:
        raise KeyError(
            f"unknown algorithm {algorithm!r}; available: {', '.join(list_kernels())}"
        )
    module = importlib.import_module(f".{algorithm}", __name__)
    return tuple(module.available_backends())
