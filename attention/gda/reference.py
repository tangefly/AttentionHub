"""Readable reference implementation of the gated delta rule.

The gated delta rule is the linear-attention recurrence used by Gated DeltaNet
and by the ``linear_attention`` blocks of Qwen3.5.  For every head it keeps a
matrix-valued recurrent state ``S`` of shape ``(d_k, d_v)`` and processes one
token at a time::

    S_t = S_{t-1} * exp(g_t)                       # gated decay
    delta_t = (v_t - S_t^T k_t) * beta_t           # delta (write) rule
    S_t = S_t + k_t delta_t^T                      # rank-one update
    o_t = S_t^T q_t

``g_t <= 0`` is the log decay, ``beta_t`` the write strength (the "learning
rate" of the delta rule).  When ``g_t = 0`` and ``beta_t = 1`` this reduces to
the classic delta rule; setting ``beta_t = 0`` freezes the state.

This module intentionally uses an explicit per-token loop.  It is the oracle
that the chunked PyTorch implementation in :mod:`attention.gda.torch` is tested
against; see :mod:`attention.gda.layer` for the parameterized layer.
"""

from __future__ import annotations

import torch

__all__ = ["gated_delta_rule", "l2norm"]


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalize ``x`` along ``dim`` (FLA-compatible formulation)."""
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the gated delta rule token by token.

    Args:
        query: ``(batch, seq_len, num_heads, head_k_dim)``.
        key: Same shape as ``query``.
        value: ``(batch, seq_len, num_heads, head_v_dim)``.
        g: Log decay ``(batch, seq_len, num_heads)``; entries should be ``<= 0``.
        beta: Write strength ``(batch, seq_len, num_heads)``, usually in
            ``(0, 1)`` after a sigmoid.
        initial_state: Optional ``(batch, num_heads, head_k_dim, head_v_dim)``
            recurrent state.  It is the state *before* the first token.
        output_final_state: Also return the state after the last token.
        use_qk_l2norm_in_kernel: L2-normalize query and key per token before the
            recurrence.
        scale: Query scale, defaults to ``head_k_dim ** -0.5``.

    Returns:
        ``(output, final_state)`` where ``output`` has shape
        ``(batch, seq_len, num_heads, head_v_dim)`` and ``final_state`` is
        ``None`` unless ``output_final_state=True``.  The final state is kept in
        float32 for numerically stable incremental decoding.
    """
    if query.shape != key.shape:
        raise ValueError(f"query/key shape mismatch: {query.shape} vs {key.shape}")
    if query.shape[:3] != value.shape[:3]:
        raise ValueError(
            "query/key/value must share (batch, seq_len, num_heads), got "
            f"{query.shape[:3]} and {value.shape[:3]}"
        )
    if beta.shape != query.shape[:3] or g.shape != query.shape[:3]:
        raise ValueError(
            "g and beta must have shape (batch, seq_len, num_heads), got "
            f"{tuple(g.shape)} and {tuple(beta.shape)}"
        )

    batch_size, seq_len, num_heads, head_k_dim = key.shape
    if seq_len == 0:
        raise ValueError("seq_len must be positive")
    head_v_dim = value.shape[-1]

    # Compute the recurrence in float32 and with (B, H, T, D) layout, which
    # makes the per-token state update easy to read.
    q = query.transpose(1, 2).float()
    k = key.transpose(1, 2).float()
    v = value.transpose(1, 2).float()
    beta = beta.transpose(1, 2).float()
    decay = g.transpose(1, 2).float()

    if use_qk_l2norm_in_kernel:
        q = l2norm(q, dim=-1)
        k = l2norm(k, dim=-1)
    q = q * (head_k_dim**-0.5 if scale is None else scale)

    if initial_state is None:
        state = torch.zeros(
            batch_size,
            num_heads,
            head_k_dim,
            head_v_dim,
            dtype=torch.float32,
            device=value.device,
        )
    else:
        state = initial_state.to(device=value.device, dtype=torch.float32)

    out = torch.zeros(
        batch_size,
        num_heads,
        seq_len,
        head_v_dim,
        dtype=torch.float32,
        device=value.device,
    )

    for t in range(seq_len):
        # 1. Decay the previous state.
        state = state * decay[:, :, t].exp()[..., None, None]
        k_t = k[:, :, t]
        v_t = v[:, :, t]
        # 2. Delta rule: only write the part of v_t that the state cannot
        #    already predict from k_t.
        kv_mem = torch.einsum("bhk,bhkv->bhv", k_t, state)
        delta = (v_t - kv_mem) * beta[:, :, t].unsqueeze(-1)
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        # 3. Read the output from the *updated* state.
        out[:, :, t] = torch.einsum("bhk,bhkv->bhv", q[:, :, t], state)

    out = out.transpose(1, 2).contiguous().to(query.dtype)
    final_state = state if output_final_state else None
    return out, final_state
