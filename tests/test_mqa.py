from __future__ import annotations

import torch

from attention.mqa import mqa
from attention.sdpa import reference
from tests.utils import tolerance


def test_mqa_matches_expanded_sdpa(device, dtype):
    torch.manual_seed(0)
    batch, heads, seq, dim = 2, 4, 16, 8
    q = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)
    k = torch.randn(batch, seq, dim, device=device, dtype=dtype)
    v = torch.randn(batch, seq, dim, device=device, dtype=dtype)

    k_expanded = k.unsqueeze(1).expand(-1, heads, -1, -1)
    v_expanded = v.unsqueeze(1).expand(-1, heads, -1, -1)
    expected, expected_weights = reference.scaled_dot_product_attention(
        q, k_expanded, v_expanded, causal=True, return_attn=True
    )

    output, weights = mqa(q, k, v, causal=True, return_attn=True)

    assert torch.allclose(output, expected, atol=tolerance(dtype), rtol=1e-3)
    assert torch.allclose(
        weights, expected_weights, atol=tolerance(dtype), rtol=1e-3
    )


def test_mqa_accepts_3d_query(device, dtype):
    torch.manual_seed(0)
    heads, seq, dim = 4, 16, 8
    q = torch.randn(heads, seq, dim, device=device, dtype=dtype)
    k = torch.randn(seq, dim, device=device, dtype=dtype)
    v = torch.randn(seq, dim, device=device, dtype=dtype)

    output, weights = mqa(q, k, v, causal=True, return_attn=True)

    assert output.shape == (heads, seq, dim)
    assert weights.shape == (heads, seq, seq)


def test_mqa_mask_is_applied(device, dtype):
    torch.manual_seed(0)
    q = torch.randn(1, 2, 8, 4, device=device, dtype=dtype)
    k = torch.randn(1, 8, 4, device=device, dtype=dtype)
    v = torch.randn(1, 8, 4, device=device, dtype=dtype)
    mask = torch.ones(8, 8, device=device, dtype=torch.bool).tril()

    _, weights = mqa(q, k, v, mask=mask, return_attn=True)

    assert torch.count_nonzero(weights[..., 0, 1:]) == 0
