"""Pure-PyTorch gated delta rule backends.

This module is a dependency-free port of the eager fallbacks that
``transformers`` ships for Qwen3.5 / Qwen3-Next::

    transformers.models.qwen3_5.modeling_qwen3_5.torch_chunk_gated_delta_rule
    transformers.models.qwen3_5.modeling_qwen3_5.torch_recurrent_gated_delta_rule

Keeping the code aligned with those functions -- same tensor layout, same
float32 internal precision, same UT transform, same padding and dtype
handling, same defaults -- makes the outputs match the ``transformers``
fallback bit for bit (see ``tests/test_gda.py``), while the module remains
usable without ``transformers`` installed.

Two entry points:

* :func:`torch_chunk_gated_delta_rule` -- the chunked / UT-transform
  formulation.  It parallelizes the intra-chunk part with matmuls and only
  scans over chunks, so it is much faster than a token loop on GPU while
  remaining plain PyTorch.
* :func:`torch_recurrent_gated_delta_rule` -- the literal per-token
  recurrence.  It is the natural choice for one-token incremental decoding and
  is kept here so the layer can share the same code path on CPU and GPU.

The short aliases :func:`gated_delta_rule` and
:func:`recurrent_gated_delta_rule` are what the rest of this package imports.
Both follow the same tensor layout as
:func:`attention.gda.reference.gated_delta_rule`.

Deliberate differences from ``transformers``
--------------------------------------------
* ``scale`` is an optional keyword.  Its default reproduces the
  ``transformers`` behaviour exactly (``head_k_dim ** -0.5``; the recurrent
  kernel uses the equivalent ``query / sqrt(head_k_dim)``).  Passing an
  explicit ``scale`` is an extension of this package.
* ``**kwargs`` are accepted and ignored, mirroring the ``transformers``
  signature (the FLA hub kernels consume entries such as ``cu_seqlens``; the
  eager path has nothing to do with them).  This keeps call sites written
  against ``transformers`` working unchanged.
* Shapes are validated up front with :func:`_check_inputs`.  This only turns
  failures that would otherwise happen deep inside a matmul into clear errors;
  valid inputs are unaffected.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .reference import l2norm

__all__ = [
    "gated_delta_rule",
    "recurrent_gated_delta_rule",
    "torch_chunk_gated_delta_rule",
    "torch_recurrent_gated_delta_rule",
]


def _is_torchdynamo_exporting() -> bool:
    """Return whether we are tracing for ``torch.export``.

    ``transformers`` falls back to an explicit forward substitution for the
    unit lower-triangular solve on export targets that do not support
    ``torch.linalg.solve_triangular``.  We mirror that check here (guarded, so
    older PyTorch builds simply take the fast path).
    """
    try:
        return torch.compiler.is_exporting()
    except Exception:  # pragma: no cover - depends on the PyTorch build
        return False


def _check_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> None:
    """Validate the shapes shared by both kernels.

    ``transformers`` assumes these shapes without checking them; validating
    here does not change any result, it only reports misuse early.
    """
    if query.shape != key.shape:
        raise ValueError(f"query/key shape mismatch: {query.shape} vs {key.shape}")
    if query.shape[:3] != value.shape[:3]:
        raise ValueError(
            "query/key/value must share (batch, seq_len, num_heads), got "
            f"{query.shape[:3]} and {value.shape[:3]}"
        )
    if query.shape[1] == 0:
        raise ValueError("seq_len must be positive")
    if g.shape != query.shape[:3] or beta.shape != query.shape[:3]:
        raise ValueError(
            "g and beta must have shape (batch, seq_len, num_heads), got "
            f"{tuple(g.shape)} and {tuple(beta.shape)}"
        )


def torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    scale: float | None = None,
    **kwargs: object,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Gated delta rule via the chunked UT-transform algorithm.

    The sequence is split into chunks of ``chunk_size``.  Inside a chunk the
    delta-rule updates are condensed into a triangular system, which turns the
    token-by-token recurrence into a handful of matmuls; chunks themselves are
    processed by a short sequential scan over the recurrent state.

    Args:
        query: ``(batch, seq_len, num_heads, head_k_dim)``.
        key: Same shape as ``query``.
        value: ``(batch, seq_len, num_heads, head_v_dim)``.  ``num_heads`` must
            equal the one of ``query``/``key``; grouped value heads are
            expanded by the caller (``repeat_interleave``), exactly like
            ``transformers``' ``Qwen3_5GatedDeltaNet``.
        g: Log decay ``(batch, seq_len, num_heads)``.
        beta: Write strength ``(batch, seq_len, num_heads)``.
        chunk_size: Chunk length; the sequence is right-padded internally.
        initial_state: Optional ``(batch, num_heads, head_k_dim, head_v_dim)``.
        output_final_state: Also return the state after the last token.
        use_qk_l2norm_in_kernel: L2-normalize query and key.
        scale: Query scale, defaults to ``head_k_dim ** -0.5``.
        **kwargs: Ignored.  Accepted for signature compatibility with the
            ``transformers`` / FLA kernels (e.g. ``cu_seqlens``).

    Returns:
        ``(output, final_state)``; see
        :func:`attention.gda.reference.gated_delta_rule`.
    """
    del kwargs  # unused by the eager path, see the module docstring

    _check_inputs(query, key, value, g, beta)
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    initial_dtype = query.dtype
    batch_size, sequence_length, _, head_k_dim = key.shape
    num_heads, head_v_dim = value.shape[-2:]
    recurrent_state_shape = (batch_size, num_heads, head_k_dim, head_v_dim)

    # Work in float32 with a (B, H, T, D) layout.
    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, g)
    ]
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    # ``transformers`` always rescales by the head dimension; ``scale`` is the
    # optional hook of this package and defaults to the same value.
    scaling = head_k_dim**-0.5 if scale is None else scale
    query = query * scaling

    # Right-pad the sequence to a multiple of the chunk size.
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query, key, value = (F.pad(x, (0, 0, 0, pad_size)) for x in (query, key, value))
    beta, decay = (F.pad(x, (0, pad_size)) for x in (beta, decay))

    total_sequence_length = sequence_length + pad_size
    num_chunks = total_sequence_length // chunk_size

    # The delta rule writes beta * (v - k^T S), so fold beta into k and v.
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    # (B, H, T, D) -> (B, H, num_chunks, chunk_size, D)
    query, key, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, k_beta, v_beta)
    ]
    decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk_size)

    # Strictly upper-triangular mask: positions a causal chunk must not see.
    strictly_upper_mask = torch.ones(
        chunk_size, chunk_size, dtype=torch.bool, device=query.device
    ).triu(1)

    # Cumulative log decay inside each chunk: cum_decay[..., t] is the log of
    # the product of decays from the start of the chunk to position t.
    cum_decay = decay.cumsum(dim=3)

    # Pairwise decay exp(cum_decay_i - cum_decay_j) between positions i >= j.
    pairwise_decay = cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)
    pairwise_decay = pairwise_decay.masked_fill(strictly_upper_mask, float("-inf"))
    pairwise_decay = pairwise_decay.exp()

    # UT transform: (I + tril(diag(beta) K K^T * decay))^{-1}.  ``new_values``
    # are the delta-rule targets ("u" in the DeltaNet paper) and
    # ``k_cumdecay`` gathers the keys that read the previous chunk state.
    ut_system = (k_beta @ key.transpose(-1, -2)) * pairwise_decay
    intra_chunk_attn = (query @ key.transpose(-1, -2)) * pairwise_decay
    decayed_k_beta = k_beta * cum_decay.exp().unsqueeze(-1)

    if not _is_torchdynamo_exporting():
        # Unit lower-triangular system: solve directly with LAPACK.
        new_values = torch.linalg.solve_triangular(
            ut_system, v_beta, upper=False, unitriangular=True
        )
        k_cumdecay = torch.linalg.solve_triangular(
            ut_system, decayed_k_beta, upper=False, unitriangular=True
        )
    else:
        # ``torch.export`` does not support the fast triangular solver, so build
        # the inverse by forward substitution instead (same result, slower).
        ut_system = -ut_system.tril(-1)  # keep only the strictly lower triangle
        for i in range(1, chunk_size):
            row = ut_system[..., i, :i].clone()
            sub = ut_system[..., :i, :i].clone()
            ut_system[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
        ut_system = ut_system + torch.eye(
            chunk_size, dtype=ut_system.dtype, device=ut_system.device
        )
        new_values = ut_system @ v_beta
        k_cumdecay = ut_system @ decayed_k_beta

    if initial_state is None:
        recurrent_state = torch.zeros(
            recurrent_state_shape,
            dtype=new_values.dtype,
            device=new_values.device,
        )
    else:
        recurrent_state = initial_state.to(new_values)

    core_attn_out = torch.zeros_like(new_values)

    # Fold the within-chunk decay into q and k once, instead of per chunk.
    query = query * cum_decay.exp().unsqueeze(-1)
    key = key * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
    chunk_decay = cum_decay[..., -1].exp()[..., None, None]

    # Sequential scan over chunks.
    for i in range(num_chunks):
        # Output: read of the previous state plus within-chunk attention.
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ recurrent_state
        inter_chunk_attn = query[:, :, i] @ recurrent_state
        core_attn_out[:, :, i] = (
            inter_chunk_attn + intra_chunk_attn[:, :, i] @ v_new
        )
        # S_{t+1} = S_t * decay + K^T v_new
        recurrent_state = (
            recurrent_state * chunk_decay[:, :, i]
            + key[:, :, i].transpose(-1, -2) @ v_new
        )

    final_state = recurrent_state if output_final_state else None

    # Flatten the chunk dimension, drop the padding and restore (B, T, H, D).
    core_attn_out = core_attn_out.reshape(
        batch_size, num_heads, -1, head_v_dim
    )[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).to(
        initial_dtype, memory_format=torch.contiguous_format
    )
    return core_attn_out, final_state


def torch_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    scale: float | None = None,
    **kwargs: object,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Vectorized per-token gated delta rule (used for incremental decoding).

    Same arguments and return value as
    :func:`torch_chunk_gated_delta_rule`, without ``chunk_size``.
    """
    del kwargs  # unused by the eager path, see the module docstring

    _check_inputs(query, key, value, g, beta)

    initial_dtype = query.dtype
    batch_size, sequence_length, _, head_k_dim = key.shape
    num_heads, head_v_dim = value.shape[-2:]
    recurrent_state_shape = (batch_size, num_heads, head_k_dim, head_v_dim)

    # Work in float32 with a (B, H, T, D) layout, exactly like ``transformers``.
    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, g)
    ]
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    # ``transformers`` divides by sqrt(head_k_dim); keep that exact form when no
    # explicit ``scale`` is given so the outputs agree bit for bit.
    query = query / (head_k_dim**0.5) if scale is None else query * scale

    if initial_state is None:
        recurrent_state = torch.zeros(
            recurrent_state_shape, dtype=value.dtype, device=value.device
        )
    else:
        recurrent_state = initial_state.to(value)

    core_attn_out = torch.zeros_like(value)
    for t in range(sequence_length):
        q_t, k_t, v_t = query[:, :, t], key[:, :, t], value[:, :, t]
        # 1. Decay the previous state.
        recurrent_state = recurrent_state * decay[:, :, t].exp()[..., None, None]
        # 2. Delta rule: only write the part of v_t the state cannot predict.
        kv_mem = (recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta[:, :, t].unsqueeze(-1)
        recurrent_state = (
            recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        )
        # 3. Read the output from the *updated* state.
        core_attn_out[:, :, t] = (recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    final_state = recurrent_state if output_final_state else None
    core_attn_out = core_attn_out.transpose(1, 2).to(
        initial_dtype, memory_format=torch.contiguous_format
    )
    return core_attn_out, final_state


#: Short names used throughout this package.
gated_delta_rule = torch_chunk_gated_delta_rule
recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule
