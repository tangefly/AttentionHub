from __future__ import annotations

import pytest
import torch

from attention.gqa import gqa
from attention.sdpa import reference
from tests.utils import tolerance


def test_gqa_matches_expanded_sdpa(device, dtype):
    torch.manual_seed(0)
    batch, q_heads, kv_heads, seq, dim = 2, 4, 2, 16, 8
    q = torch.randn(batch, q_heads, seq, dim, device=device, dtype=dtype)
    k = torch.randn(batch, kv_heads, seq, dim, device=device, dtype=dtype)
    v = torch.randn(batch, kv_heads, seq, dim, device=device, dtype=dtype)

    group_size = q_heads // kv_heads
    k_expanded = k.repeat_interleave(group_size, dim=1)
    v_expanded = v.repeat_interleave(group_size, dim=1)
    expected, expected_weights = reference.scaled_dot_product_attention(
        q, k_expanded, v_expanded, causal=True, return_attn=True
    )

    output, weights = gqa(q, k, v, causal=True, return_attn=True)

    assert torch.allclose(output, expected, atol=tolerance(dtype), rtol=1e-3)
    assert torch.allclose(
        weights, expected_weights, atol=tolerance(dtype), rtol=1e-3
    )


def test_gqa_attention_weight_shape(device, dtype):
    q = torch.randn(2, 8, 16, 8, device=device, dtype=dtype)
    k = torch.randn(2, 4, 16, 8, device=device, dtype=dtype)
    v = torch.randn(2, 4, 16, 8, device=device, dtype=dtype)

    _, weights = gqa(q, k, v, return_attn=True)

    assert weights.shape == (2, 8, 16, 16)


def test_gqa_rejects_invalid_head_ratio(device, dtype):
    q = torch.randn(1, 3, 8, 4, device=device, dtype=dtype)
    k = torch.randn(1, 2, 8, 4, device=device, dtype=dtype)
    v = torch.randn(1, 2, 8, 4, device=device, dtype=dtype)

    with pytest.raises(ValueError, match="divisible"):
        gqa(q, k, v)
