from __future__ import annotations

import pytest
import torch

from attention.moba import available_backends, moba
from attention.sdpa import reference as sdpa_reference
from tests.utils import tolerance


def _dense_causal(q, k, v, cu_seqlens):
    outputs = []
    for index in range(cu_seqlens.numel() - 1):
        start = int(cu_seqlens[index].item())
        end = int(cu_seqlens[index + 1].item())
        q_i = q[start:end].transpose(0, 1).unsqueeze(0)
        k_i = k[start:end].transpose(0, 1).unsqueeze(0)
        v_i = v[start:end].transpose(0, 1).unsqueeze(0)
        out_i, _ = sdpa_reference.scaled_dot_product_attention(
            q_i, k_i, v_i, causal=True, return_attn=True
        )
        outputs.append(out_i.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0)


def test_moba_with_all_blocks_equals_causal_attention(device, dtype):
    torch.manual_seed(0)
    total, heads, head_dim = 32, 2, 8
    q = torch.randn(total, heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(total, heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(total, heads, head_dim, device=device, dtype=dtype)
    cu_seqlens = torch.tensor([0, 20, total], device=device, dtype=torch.int32)

    expected = _dense_causal(q, k, v, cu_seqlens)
    output = moba(
        q,
        k,
        v,
        cu_seqlens,
        max_seqlen=20,
        moba_chunk_size=8,
        moba_topk=4,
        backend="reference",
    )

    assert torch.allclose(output, expected, atol=tolerance(dtype), rtol=1e-3)


def test_moba_auto_uses_available_backend(device, dtype):
    torch.manual_seed(0)
    total, heads, head_dim = 16, 2, 8
    q = torch.randn(total, heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(total, heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(total, heads, head_dim, device=device, dtype=dtype)
    cu_seqlens = torch.tensor([0, 8, total], device=device, dtype=torch.int32)

    expected_backend = "flash" if "flash" in available_backends() else "reference"
    output = moba(
        q,
        k,
        v,
        cu_seqlens,
        max_seqlen=8,
        moba_chunk_size=8,
        moba_topk=2,
    )
    expected = moba(
        q,
        k,
        v,
        cu_seqlens,
        max_seqlen=8,
        moba_chunk_size=8,
        moba_topk=2,
        backend=expected_backend,
    )

    assert torch.allclose(output, expected, atol=tolerance(dtype), rtol=1e-3)


def test_moba_available_backends():
    assert "reference" in available_backends()


def test_moba_invalid_backend(device, dtype):
    q = torch.randn(8, 1, 4, device=device, dtype=dtype)
    cu_seqlens = torch.tensor([0, 8], device=device, dtype=torch.int32)
    with pytest.raises(ValueError, match="not available"):
        moba(
            q,
            q,
            q,
            cu_seqlens,
            max_seqlen=8,
            moba_chunk_size=8,
            moba_topk=1,
            backend="triton",
        )


@pytest.mark.flash
def test_moba_flash_matches_reference(device, dtype):
    if "flash" not in available_backends():
        pytest.skip("flash-attn with CUDA is not available")

    torch.manual_seed(0)
    total, heads, head_dim = 64, 2, 16
    q = torch.randn(total, heads, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(total, heads, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(total, heads, head_dim, device=device, dtype=torch.float16)
    cu_seqlens = torch.tensor([0, 32, total], device=device, dtype=torch.int32)

    expected = moba(
        q,
        k,
        v,
        cu_seqlens,
        max_seqlen=32,
        moba_chunk_size=16,
        moba_topk=2,
        backend="reference",
    )
    output = moba(
        q,
        k,
        v,
        cu_seqlens,
        max_seqlen=32,
        moba_chunk_size=16,
        moba_topk=2,
        backend="flash",
    )

    assert torch.allclose(output, expected, atol=2e-2, rtol=1e-2)
