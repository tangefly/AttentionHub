from __future__ import annotations

import pytest
import torch

from attention.norm import QwenRMSNorm, RMSNorm


def _manual_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.float()
    normalized = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (normalized * weight.float()).type_as(x)


def test_rms_norm_matches_manual(device):
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8, 16, device=device)
    norm = RMSNorm(16, eps=1e-6).to(device)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(16, device=device))

    expected = _manual_rms_norm(x, norm.weight, norm.eps)

    assert torch.allclose(norm(x), expected, atol=1e-6, rtol=1e-5)


def test_rms_norm_default_weight_is_identity_scale(device):
    torch.manual_seed(0)
    x = torch.randn(4, 16, device=device)
    norm = RMSNorm(16, eps=1e-6).to(device)

    normalized = x.float() * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.eps)

    assert torch.allclose(norm(x), normalized.type_as(x), atol=1e-6, rtol=1e-5)


def test_qwen_rms_norm_uses_unit_offset(device):
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8, 16, device=device)
    norm = QwenRMSNorm(16, eps=1e-6).to(device)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(16, device=device))

    expected = _manual_rms_norm(x, 1.0 + norm.weight, norm.eps)

    assert torch.allclose(norm(x), expected, atol=1e-6, rtol=1e-5)


def test_qwen_rms_norm_zero_init_is_identity_scale(device):
    torch.manual_seed(0)
    x = torch.randn(4, 16, device=device)
    norm = QwenRMSNorm(16, eps=1e-6).to(device)

    normalized = x.float() * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.eps)

    assert torch.equal(norm.weight, torch.zeros(16, device=device))
    assert torch.allclose(norm(x), normalized.type_as(x), atol=1e-6, rtol=1e-5)


def test_grouped_rms_norm_matches_manual(device):
    torch.manual_seed(0)
    x = torch.randn(2, 3, 32, device=device)
    norm = QwenRMSNorm(32, group_size=8, eps=1e-6).to(device)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(32, device=device))

    grouped = x.float().reshape(*x.shape[:-1], -1, 8)
    normalized = grouped * torch.rsqrt(
        grouped.pow(2).mean(-1, keepdim=True) + norm.eps
    )
    expected = normalized.flatten(-2) * (1.0 + norm.weight).float()

    assert torch.allclose(norm(x), expected, atol=1e-6, rtol=1e-5)


def test_rms_norm_preserves_input_dtype(device):
    torch.manual_seed(0)
    x = torch.randn(2, 16, device=device, dtype=torch.float16)
    norm = RMSNorm(16, eps=1e-6).to(device, dtype=torch.float16)

    output = norm(x)

    assert output.dtype == x.dtype
    assert torch.allclose(output.float(), norm(x.float()).float(), atol=1e-2, rtol=1e-2)


def test_rms_norm_rejects_invalid_group_size():
    with pytest.raises(ValueError, match="divisible"):
        RMSNorm(32, group_size=7)


def test_qsa_reuses_shared_qwen_rms_norm():
    from attention.qsa.layer import RMSNorm as QSARMNorm

    assert QSARMNorm is QwenRMSNorm


def test_norm_top_level_exports():
    import attention

    assert attention.RMSNorm is RMSNorm
    assert attention.QwenRMSNorm is QwenRMSNorm
