from __future__ import annotations

import pytest
import torch

from attention.rope import (
    apply_rotary_pos_emb,
    build_rope_cache,
    build_rope_cache_batched,
)


def test_rope_cache_shape_and_identity(device):
    cos, sin = build_rope_cache(16, 8, device=device)

    assert cos.shape == (16, 8)
    assert sin.shape == (16, 8)
    assert torch.allclose(
        cos**2 + sin**2,
        torch.ones_like(cos),
        atol=1e-6,
    )


def test_rope_cache_first_position(device):
    cos, sin = build_rope_cache(4, 8, device=device)

    assert torch.equal(cos[0], torch.ones(8, device=device))
    assert torch.equal(sin[0], torch.zeros(8, device=device))


def test_rope_cache_batched_matches_expanded(device):
    cos, sin = build_rope_cache(12, 8, device=device)
    cos_batched, sin_batched = build_rope_cache_batched(3, 12, 8, device=device)

    assert cos_batched.shape == (3, 12, 8)
    assert torch.allclose(cos_batched, cos.unsqueeze(0).expand(3, -1, -1))
    assert torch.allclose(sin_batched, sin.unsqueeze(0).expand(3, -1, -1))


def test_apply_rotary_preserves_norm(device):
    torch.manual_seed(0)
    q = torch.randn(2, 4, 16, 8, device=device)
    k = torch.randn(2, 4, 16, 8, device=device)
    cos, sin = build_rope_cache_batched(2, 16, 8, device=device)

    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    assert torch.allclose(q_rot.norm(dim=-1), q.norm(dim=-1), atol=1e-5)
    assert torch.allclose(k_rot.norm(dim=-1), k.norm(dim=-1), atol=1e-5)


def test_partial_rope_leaves_non_rotary_part_unchanged(device):
    torch.manual_seed(0)
    q = torch.randn(1, 2, 4, 8, device=device)
    cos, sin = build_rope_cache_batched(1, 4, 4, device=device)

    q_rot = apply_rotary_pos_emb(q, cos=cos, sin=sin)

    assert isinstance(q_rot, torch.Tensor)
    assert torch.equal(q_rot[..., 4:], q[..., 4:])


def test_apply_rotary_requires_cos_and_sin(device):
    q = torch.randn(1, 2, 4, 8, device=device)
    with pytest.raises(ValueError, match="cos and sin"):
        apply_rotary_pos_emb(q)


def test_apply_rotary_rejects_mismatched_cache(device):
    q = torch.randn(1, 2, 4, 8, device=device)
    cos, _ = build_rope_cache_batched(1, 4, 8, device=device)
    sin, _ = build_rope_cache_batched(1, 4, 4, device=device)
    with pytest.raises(ValueError, match="same shape"):
        apply_rotary_pos_emb(q, cos=cos, sin=sin)


def test_rope_cache_rejects_odd_dim(device):
    with pytest.raises(ValueError, match="even"):
        build_rope_cache(4, 7, device=device)


def test_rope_cache_dtype(device):
    cos, sin = build_rope_cache(4, 8, device=device, dtype=torch.float16)
    assert cos.dtype == torch.float16
    assert sin.dtype == torch.float16


def test_rope_top_level_exports():
    import attention

    assert callable(attention.build_rope_cache)
    assert callable(attention.apply_rotary_pos_emb)
    assert attention.rope.build_rope_cache is build_rope_cache
