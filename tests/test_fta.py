from __future__ import annotations

import pytest
import torch

from attention.fta import fta


def test_fta_output_shape(device, dtype):
    q = torch.randn(2, 2, 16, 8, device=device, dtype=dtype)
    k = torch.randn(2, 2, 16, 8, device=device, dtype=dtype)
    v = torch.randn(2, 2, 16, 8, device=device, dtype=dtype)

    output = fta(q, k, v, l=2)

    assert output.shape == q.shape
    assert torch.isfinite(output).all()


def test_fta_is_causal(device, dtype):
    torch.manual_seed(0)
    q = torch.randn(1, 2, 12, 8, device=device, dtype=dtype)
    k = torch.randn(1, 2, 12, 8, device=device, dtype=dtype)
    v = torch.randn(1, 2, 12, 8, device=device, dtype=dtype)
    v_changed = v.clone()
    v_changed[:, :, -1] += 10.0

    output = fta(q, k, v, l=2)
    output_changed = fta(q, k, v_changed, l=2)

    assert torch.allclose(output[:, :, :-1], output_changed[:, :, :-1])


def test_fta_rejects_odd_head_dim(device, dtype):
    q = torch.randn(1, 1, 8, 7, device=device, dtype=dtype)
    with pytest.raises(ValueError, match="divisible"):
        fta(q, q, q, l=2)


def test_fta_rejects_invalid_window(device, dtype):
    q = torch.randn(1, 1, 8, 8, device=device, dtype=dtype)
    with pytest.raises(ValueError, match=">= 1"):
        fta(q, q, q, l=0)


def test_fta_return_attn_not_supported(device, dtype):
    q = torch.randn(1, 1, 8, 8, device=device, dtype=dtype)
    with pytest.raises(ValueError, match="return_attn"):
        fta(q, q, q, l=2, return_attn=True)
