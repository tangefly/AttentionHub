from __future__ import annotations

import pytest
import torch

from attention.sdpa import reference, sdpa
from tests.utils import tolerance


def _inputs(device, dtype):
    torch.manual_seed(0)
    q = torch.randn(2, 4, 16, 8, device=device, dtype=dtype)
    k = torch.randn(2, 4, 16, 8, device=device, dtype=dtype)
    v = torch.randn(2, 4, 16, 8, device=device, dtype=dtype)
    return q, k, v


@pytest.mark.parametrize("causal", [False, True])
def test_torch_matches_reference(device, dtype, causal):
    q, k, v = _inputs(device, dtype)
    out_ref, weights_ref = sdpa(
        q, k, v, causal=causal, backend="reference", return_attn=True
    )
    out_torch = sdpa(q, k, v, causal=causal, backend="torch")

    assert torch.allclose(out_torch, out_ref, atol=tolerance(dtype), rtol=1e-3)
    assert weights_ref.shape == (2, 4, 16, 16)


def test_default_returns_output_only(device, dtype):
    q, k, v = _inputs(device, dtype)
    out = sdpa(q, k, v)
    assert isinstance(out, torch.Tensor)
    assert out.shape == q.shape


def test_bool_mask_uses_true_as_keep(device, dtype):
    q, k, v = _inputs(device, dtype)
    mask = torch.ones(16, 16, device=device, dtype=torch.bool).tril()

    _, weights = sdpa(
        q, k, v, mask=mask, backend="reference", return_attn=True
    )
    # Query 0 may only attend to key 0.
    assert torch.count_nonzero(weights[..., 0, 1:]) == 0


def test_float_mask_matches_bool_mask(device, dtype):
    q, k, v = _inputs(device, dtype)
    bool_mask = torch.ones(16, 16, device=device, dtype=torch.bool).tril()
    float_mask = torch.zeros(16, 16, device=device, dtype=dtype).masked_fill(
        ~bool_mask, torch.finfo(dtype).min
    )

    _, weights_bool = sdpa(
        q, k, v, mask=bool_mask, backend="reference", return_attn=True
    )
    _, weights_float = sdpa(
        q, k, v, mask=float_mask, backend="reference", return_attn=True
    )

    assert torch.allclose(
        weights_float, weights_bool, atol=tolerance(dtype), rtol=1e-3
    )


def test_explicit_scale_and_dropout(device, dtype):
    q, k, v = _inputs(device, dtype)
    out = sdpa(q, k, v, scale=0.5, dropout_p=0.0, backend="reference")
    assert out.shape == q.shape


def test_invalid_backend_raises(device, dtype):
    q, k, v = _inputs(device, dtype)
    with pytest.raises(ValueError, match="not available"):
        sdpa(q, k, v, backend="cuda")


def test_return_attn_with_torch_backend_raises(device, dtype):
    q, k, v = _inputs(device, dtype)
    with pytest.raises(ValueError, match="does not materialize"):
        sdpa(q, k, v, backend="torch", return_attn=True)


def test_reference_accepts_2d_inputs(device):
    q = torch.randn(8, 8, device=device)
    k = torch.randn(8, 8, device=device)
    v = torch.randn(8, 8, device=device)
    out, weights = reference.scaled_dot_product_attention(q, k, v, causal=True)
    assert out.shape == (8, 8)
    assert weights.shape == (8, 8)
