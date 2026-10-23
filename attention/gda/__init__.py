"""Gated delta attention (GDA).

GDA -- the gated delta rule used by Gated DeltaNet and by Qwen3.5's linear
attention blocks -- is a linear-attention recurrence with a matrix state, a
per-token log decay ``g`` and a delta-rule write strength ``beta``::

    S_t = S_{t-1} * exp(g_t)
    S_t = S_t + beta_t * k_t (v_t - S_t^T k_t)^T
    o_t = S_t^T q_t

This package provides:

* :func:`gated_delta_rule` -- backend-selecting function (``reference`` or the
  chunked ``torch`` implementation);
* :func:`recurrent_gated_delta_rule` -- token-by-token kernel for decoding;
* :class:`GatedDeltaNet` -- the parameterized Qwen3.5 block (causal conv,
  decay/gate projections, gated RMSNorm, output projection) with an optional
  incremental-decoding state.
"""

from __future__ import annotations

from torch import Tensor

from ..dispatch import resolve_backend
from . import reference
from . import torch as torch_backend
from .layer import (
    GatedDeltaNet,
    GatedDeltaNetState,
    GatedRMSNorm,
    Qwen3_5GatedDeltaNet,
    Qwen3_5RMSNormGated,
)
from .torch import recurrent_gated_delta_rule

_BACKENDS = {
    "reference": reference.gated_delta_rule,
    "torch": torch_backend.gated_delta_rule,
}

#: Descriptive aliases.
GatedDeltaAttention = GatedDeltaNet


def available_backends() -> tuple[str, ...]:
    return ("reference", "torch")


def gated_delta_rule(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    *,
    chunk_size: int = 64,
    initial_state: Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    scale: float | None = None,
    backend: str = "auto",
) -> tuple[Tensor, Tensor | None]:
    """Compute the gated delta rule.

    Args:
        query, key: ``(batch, seq_len, num_heads, head_k_dim)``.
        value: ``(batch, seq_len, num_heads, head_v_dim)``.
        g: Log decay ``(batch, seq_len, num_heads)``; entries should be
            ``<= 0``.
        beta: Write strength ``(batch, seq_len, num_heads)``.
        chunk_size: Chunk length for the ``torch`` backend.
        initial_state: Optional recurrent state before the first token.
        output_final_state: Also return the state after the last token.
        use_qk_l2norm_in_kernel: L2-normalize query/key per token.
        scale: Query scale, defaults to ``head_k_dim ** -0.5``.
        backend: ``"auto"``, ``"reference"`` or ``"torch"``.

    Returns:
        ``(output, final_state)``; ``final_state`` is ``None`` unless
        ``output_final_state=True``.  See
        :func:`attention.gda.reference.gated_delta_rule` for the exact
        recurrence.
    """
    name = resolve_backend(backend, available_backends(), "gda")
    if name == "reference":
        return reference.gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            scale=scale,
        )
    return torch_backend.gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        chunk_size=chunk_size,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        scale=scale,
    )


__all__ = [
    "GatedDeltaAttention",
    "GatedDeltaNet",
    "GatedDeltaNetState",
    "GatedRMSNorm",
    "Qwen3_5GatedDeltaNet",
    "Qwen3_5RMSNormGated",
    "available_backends",
    "gated_delta_rule",
    "recurrent_gated_delta_rule",
]
