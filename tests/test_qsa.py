from __future__ import annotations

import math

import torch

from attention.qsa import QwenSparseAttention
from attention.qsa.layer import repeat_kv
from attention.rope import apply_rotary_pos_emb, build_rope_cache_batched


def _make_layer(seq_len: int) -> QwenSparseAttention:
    return QwenSparseAttention(
        head_dim=32,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_dropout=0.0,
        attention_bias=False,
        rms_norm_eps=1e-6,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=seq_len,
        indexer_compress_ratio=8,
    ).eval()


def _causal_mask(batch: int, seq_len: int, device, dtype):
    causal = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).tril()
    mask = torch.zeros(batch, 1, seq_len, seq_len, device=device, dtype=dtype)
    return mask.masked_fill(~causal, torch.finfo(dtype).min)


@torch.no_grad()
def test_qsa_full_budget_matches_dense(device):
    torch.manual_seed(0)
    batch, seq_len = 2, 32
    layer = _make_layer(seq_len).to(device)
    hidden = torch.randn(batch, seq_len, 64, device=device)
    cos, sin = build_rope_cache_batched(batch, seq_len, 16, device=device)
    mask = _causal_mask(batch, seq_len, device, torch.float32)

    output, weights = layer(hidden, (cos, sin), mask)

    q, gate = torch.chunk(
        layer.q_proj(hidden).view(batch, seq_len, -1, layer.head_dim * 2),
        2,
        dim=-1,
    )
    gate = gate.reshape(batch, seq_len, -1)
    q = layer.q_norm(q.view(batch, seq_len, 4, layer.head_dim)).transpose(1, 2)
    k = layer.k_norm(
        layer.k_proj(hidden).view(batch, seq_len, 2, layer.head_dim)
    ).transpose(1, 2)
    v = layer.v_proj(hidden).view(batch, seq_len, 2, layer.head_dim).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    k, v = repeat_kv(k, 2), repeat_kv(v, 2)

    scores = q @ k.transpose(2, 3) / math.sqrt(layer.head_dim) + mask
    probs = torch.softmax(scores, dim=-1)
    dense = (probs @ v).transpose(1, 2).reshape(batch, seq_len, -1)
    dense = layer.o_proj(dense * torch.sigmoid(gate))

    assert torch.allclose(output, dense, atol=1e-5, rtol=1e-4)
    assert torch.allclose(
        weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-5
    )


@torch.no_grad()
def test_qsa_bool_and_float_masks_are_equivalent(device):
    torch.manual_seed(0)
    batch, seq_len = 2, 32
    layer = _make_layer(seq_len).to(device)
    hidden = torch.randn(batch, seq_len, 64, device=device)
    cos, sin = build_rope_cache_batched(batch, seq_len, 16, device=device)
    float_mask = _causal_mask(batch, seq_len, device, torch.float32)
    bool_mask = float_mask == 0

    output_float, _ = layer(hidden, (cos, sin), float_mask)
    output_bool, _ = layer(hidden, (cos, sin), bool_mask)

    assert torch.allclose(output_float, output_bool, atol=1e-6, rtol=1e-5)

