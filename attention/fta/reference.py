"""Reference implementation of factorized attention."""

from __future__ import annotations

import torch
from torch import Tensor


def _build_stride_mask(num: int, stride: int, device: torch.device | str) -> Tensor:
    """Return a mask that keeps every ``stride``-th causal position."""
    idx = torch.arange(num, device=device)
    i = idx[:, None]
    j = idx[None, :]
    future = j > i
    not_stride = (i - j) % stride != 0
    return future | not_stride


def factorized_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    l: int,
) -> Tensor:
    """Compute factorized attention.

    One half of the head dimension attends locally within a window of ``l``
    tokens, the other half attends to every ``l``-th causal position.

    Args:
        Q: Query tensor of shape ``(batch, head, seq, dim)``.
        K: Key tensor of shape ``(batch, head, seq, dim)``.
        V: Value tensor of shape ``(batch, head, seq, dim)``.
        l: Local window size / stride.

    Returns:
        The attention output of shape ``(batch, head, seq, dim)``.
    """
    if Q.size(-1) != K.size(-1):
        raise ValueError(
            f"Q and K must have the same feature dim, got {Q.size(-1)} and {K.size(-1)}"
        )
    if K.size(-2) != V.size(-2):
        raise ValueError(
            f"K and V must have the same sequence length, got {K.size(-2)} and {V.size(-2)}"
        )
    if Q.dim() != 4:
        raise ValueError("FTA expects Q/K/V of shape (batch, head, seq, dim)")
    if l < 1:
        raise ValueError(f"l must be >= 1, got {l}")

    fact_head = 2
    batch, head, num, dim = Q.shape
    if dim % fact_head != 0:
        raise ValueError(f"head_dim ({dim}) must be divisible by {fact_head}")

    branch_dim = dim // fact_head
    Q = Q.view(batch, head, fact_head, num, branch_dim)
    K = K.view(batch, head, fact_head, num, branch_dim)
    V = V.view(batch, head, fact_head, num, branch_dim)
    scale = branch_dim**-0.5

    idx = torch.arange(num, device=Q.device)
    local_mask = (idx[None, :] > idx[:, None]) | (
        idx[None, :] < (idx[:, None] - l)
    )

    # 1. Local attention branch
    Q1, K1, V1 = Q[:, :, 0], K[:, :, 0], V[:, :, 0]
    scores1 = torch.matmul(Q1, K1.transpose(-2, -1)) * scale
    scores1 = scores1.masked_fill(local_mask, torch.finfo(scores1.dtype).min)
    weights1 = torch.softmax(scores1, dim=-1)
    output1 = torch.matmul(weights1, V1)

    # 2. Strided attention branch
    Q2, K2, V2 = Q[:, :, 1], K[:, :, 1], V[:, :, 1]
    stride_mask = _build_stride_mask(num, l, Q.device)
    scores2 = torch.matmul(Q2, K2.transpose(-2, -1)) * scale
    scores2 = scores2.masked_fill(stride_mask, torch.finfo(scores2.dtype).min)
    weights2 = torch.softmax(scores2, dim=-1)
    output2 = torch.matmul(weights2, V2)

    # 3. Merge both branches
    output = torch.stack([output1, output2], dim=-3)
    return output.reshape(batch, head, num, dim)


fta = factorized_attention
