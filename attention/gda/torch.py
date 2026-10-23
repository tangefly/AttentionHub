"""Pure-PyTorch gated delta rule backends.

Two implementations live here:

* :func:`gated_delta_rule` -- the chunked / UT-transform formulation.  It
  parallelizes the intra-chunk part with matmuls and only scans over chunks, so
  it is much faster than a token loop on GPU while remaining plain PyTorch.
* :func:`recurrent_gated_delta_rule` -- the literal per-token recurrence.  It is
  the natural choice for one-token incremental decoding and is kept here so the
  layer can share the same code path on CPU and GPU.

Both follow the same tensor layout as
:func:`attention.gda.reference.gated_delta_rule`.
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


def _check_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> None:
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


def gated_delta_rule(
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
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Gated delta rule via the chunked UT-transform algorithm.

    The sequence is split into chunks of ``chunk_size``.  Inside a chunk the
    delta-rule updates are condensed into a triangular system, which turns the
    token-by-token recurrence into a handful of matmuls; chunks themselves are
    processed by a short sequential scan over the recurrent state.

    Args:
        query: ``(batch, seq_len, num_heads, head_k_dim)``.
        key: Same shape as ``query``.
        value: ``(batch, seq_len, num_heads, head_v_dim)``.
        g: Log decay ``(batch, seq_len, num_heads)``.
        beta: Write strength ``(batch, seq_len, num_heads)``.
        chunk_size: Chunk length; the sequence is right-padded internally.
        initial_state: Optional ``(batch, num_heads, head_k_dim, head_v_dim)``.
        output_final_state: Also return the state after the last token.
        use_qk_l2norm_in_kernel: L2-normalize query and key.
        scale: Query scale, defaults to ``head_k_dim ** -0.5``.

    Returns:
        ``(output, final_state)``; see
        :func:`attention.gda.reference.gated_delta_rule`.
    """
    _check_inputs(query, key, value, g, beta)
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    initial_dtype = query.dtype
    batch_size, sequence_length, _, head_k_dim = key.shape
    num_heads, head_v_dim = value.shape[-2:]

    # Work in float32 with a (B, H, T, D) layout.
    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, g)
    ]
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1)
        key = l2norm(key, dim=-1)
    query = query * (head_k_dim**-0.5 if scale is None else scale)

    # Right-pad the sequence to a multiple of the chunk size.
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query, key, value = (F.pad(x, (0, 0, 0, pad_size)) for x in (query, key, value))
    beta, decay = (F.pad(x, (0, pad_size)) for x in (beta, decay))

    total_length = sequence_length + pad_size
    num_chunks = total_length // chunk_size

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
    strictly_upper = torch.ones(
        chunk_size, chunk_size, dtype=torch.bool, device=query.device
    ).triu(1)

    # Cumulative log decay inside each chunk.
    cum_decay = decay.cumsum(dim=3)

    # Pairwise decay exp(cum_decay_i - cum_decay_j) between positions i >= j.
    pairwise_decay = cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)
    pairwise_decay = pairwise_decay.masked_fill(strictly_upper, float("-inf")).exp()

    # UT transform: (I + tril(diag(beta) K K^T * decay))^{-1}.
    ut_system = (k_beta @ key.transpose(-1, -2)) * pairwise_decay
    intra_chunk_attn = (query @ key.transpose(-1, -2)) * pairwise_decay
    decayed_k_beta = k_beta * cum_decay.exp().unsqueeze(-1)

    # Solve the unit lower-triangular system directly.  ``new_values`` are the
    # delta-rule targets ("u" in the DeltaNet paper) and ``k_cumdecay`` gathers
    # the keys that read the previous chunk state.
    new_values = torch.linalg.solve_triangular(
        ut_system, v_beta, upper=False, unitriangular=True
    )
    k_cumdecay = torch.linalg.solve_triangular(
        ut_system, decayed_k_beta, upper=False, unitriangular=True
    )

    if initial_state is None:
        recurrent_state = torch.zeros(
            batch_size,
            num_heads,
            head_k_dim,
            head_v_dim,
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

    core_attn_out = core_attn_out.reshape(
        batch_size, num_heads, total_length, head_v_dim
    )[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).to(initial_dtype)
    return core_attn_out.contiguous(), final_state


def recurrent_gated_delta_rule(
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
    """Vectorized per-token gated delta rule (used for incremental decoding).

    Same arguments and return value as :func:`gated_delta_rule`, without
    ``chunk_size``.
    """
    _check_inputs(query, key, value, g, beta)

    initial_dtype = query.dtype
    batch_size, sequence_length, _, head_k_dim = key.shape
    num_heads, head_v_dim = value.shape[-2:]

    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, g)
    ]
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1)
        key = l2norm(key, dim=-1)
    query = query * (head_k_dim**-0.5 if scale is None else scale)

    if initial_state is None:
        recurrent_state = torch.zeros(
            batch_size,
            num_heads,
            head_k_dim,
            head_v_dim,
            dtype=torch.float32,
            device=value.device,
        )
    else:
        recurrent_state = initial_state.to(
            device=value.device, dtype=torch.float32
        )

    out = torch.zeros_like(value)
    for t in range(sequence_length):
        q_t, k_t, v_t = query[:, :, t], key[:, :, t], value[:, :, t]
        recurrent_state = recurrent_state * decay[:, :, t].exp()[..., None, None]
        kv_mem = torch.einsum("bhk,bhkv->bhv", k_t, recurrent_state)
        delta = (v_t - kv_mem) * beta[:, :, t].unsqueeze(-1)
        recurrent_state = (
            recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        )
        out[:, :, t] = torch.einsum("bhk,bhkv->bhv", q_t, recurrent_state)

    out = out.transpose(1, 2).contiguous().to(initial_dtype)
    final_state = recurrent_state if output_final_state else None
    return out, final_state


#: Compatibility names used by the original ``transformers`` module.
torch_chunk_gated_delta_rule = gated_delta_rule
torch_recurrent_gated_delta_rule = recurrent_gated_delta_rule
