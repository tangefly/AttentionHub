from __future__ import annotations

import pytest
import torch

from attention.mla import DeepseekV2Attention
from attention.rope import apply_rotary_pos_emb, build_rope_cache_batched


def _make_layer(
    q_lora_rank: int | None = 16,
    rope_layout: str = "interleaved",
) -> DeepseekV2Attention:
    return DeepseekV2Attention(
        hidden_size=32,
        num_attention_heads=4,
        kv_lora_rank=8,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        q_lora_rank=q_lora_rank,
        attention_dropout=0.0,
        attention_bias=False,
        rms_norm_eps=1e-6,
        rope_layout=rope_layout,
    ).eval()


def _causal_bool(batch: int, seq_len: int, device) -> torch.Tensor:
    causal = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).tril()
    return causal.view(1, 1, seq_len, seq_len).expand(batch, 1, seq_len, seq_len)


def _causal_float(batch: int, seq_len: int, device) -> torch.Tensor:
    mask = _causal_bool(batch, seq_len, device)
    return torch.zeros(
        batch, 1, seq_len, seq_len, device=device, dtype=torch.float32
    ).masked_fill(~mask, torch.finfo(torch.float32).min)


def _complex_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """DeepSeek's interleaved complex RoPE, used as an independent oracle."""
    rotary_dim = x.shape[-1]
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = torch.view_as_complex(
        torch.stack(
            (cos[..., : rotary_dim // 2], sin[..., : rotary_dim // 2]),
            dim=-1,
        )
    )
    return torch.view_as_real(x_complex * freqs.unsqueeze(1)).flatten(3).type_as(x)


@torch.no_grad()
def _manual_reference(
    layer: DeepseekV2Attention,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    causal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seq_len, _ = hidden_states.shape
    num_heads = layer.num_heads

    if layer.q_lora_rank is None:
        q = layer.q_proj(hidden_states)
    else:
        q = layer.q_b_proj(layer.q_a_layernorm(layer.q_a_proj(hidden_states)))
    q = q.view(batch, seq_len, num_heads, layer.qk_head_dim).transpose(1, 2)
    q_nope, q_pe = torch.split(
        q, [layer.qk_nope_head_dim, layer.qk_rope_head_dim], dim=-1
    )

    compressed_kv = layer.kv_a_proj_with_mqa(hidden_states)
    kv_nope, k_pe = torch.split(
        compressed_kv, [layer.kv_lora_rank, layer.qk_rope_head_dim], dim=-1
    )
    latent = layer.kv_a_layernorm(kv_nope).view(
        batch, 1, seq_len, layer.kv_lora_rank
    )
    k_pe = k_pe.view(batch, 1, seq_len, layer.qk_rope_head_dim)
    if layer.rope_layout == "interleaved":
        q_pe = _complex_rope(q_pe, cos, sin)
        k_pe = _complex_rope(k_pe, cos, sin)
    else:
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin)

    kv = layer.kv_b_proj(latent).view(
        batch, seq_len, num_heads, layer.qk_nope_head_dim + layer.v_head_dim
    )
    kv = kv.transpose(1, 2)
    k_nope, value_states = torch.split(
        kv, [layer.qk_nope_head_dim, layer.v_head_dim], dim=-1
    )
    key_states = torch.cat(
        [k_nope, k_pe.expand(-1, num_heads, -1, -1)], dim=-1
    )
    query_states = torch.cat([q_nope, q_pe], dim=-1)

    scores = torch.matmul(query_states, key_states.transpose(2, 3)) * layer.scaling
    scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
    attn_weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    output = torch.matmul(attn_weights, value_states)
    output = output.transpose(1, 2).reshape(batch, seq_len, -1)
    return layer.o_proj(output), attn_weights


@pytest.mark.parametrize("rope_layout", ["interleaved", "half"])
@torch.no_grad()
def test_mla_matches_manual_reference(device, rope_layout):
    torch.manual_seed(0)
    batch, seq_len = 2, 12
    layer = _make_layer(rope_layout=rope_layout).to(device)
    hidden = torch.randn(batch, seq_len, 32, device=device)
    cos, sin = build_rope_cache_batched(batch, seq_len, 4, device=device)
    causal = _causal_bool(batch, seq_len, device)

    output, weights = layer(hidden, (cos, sin), causal)
    expected_output, expected_weights = _manual_reference(
        layer, hidden, cos, sin, causal
    )

    assert output.shape == (batch, seq_len, 32)
    assert weights.shape == (batch, 4, seq_len, seq_len)
    assert torch.allclose(output, expected_output, atol=1e-6, rtol=1e-5)
    assert torch.allclose(weights, expected_weights, atol=1e-6, rtol=1e-5)


@torch.no_grad()
def test_mla_without_query_lora(device):
    torch.manual_seed(0)
    batch, seq_len = 2, 7
    layer = _make_layer(q_lora_rank=None).to(device)
    assert layer.q_proj is not None
    assert layer.q_a_proj is None
    assert layer.q_a_layernorm is None

    hidden = torch.randn(batch, seq_len, 32, device=device)
    cos, sin = build_rope_cache_batched(batch, seq_len, 4, device=device)
    causal = _causal_bool(batch, seq_len, device)

    output, weights = layer(hidden, (cos, sin), causal)
    expected_output, expected_weights = _manual_reference(
        layer, hidden, cos, sin, causal
    )

    assert torch.allclose(output, expected_output, atol=1e-6, rtol=1e-5)
    assert torch.allclose(weights, expected_weights, atol=1e-6, rtol=1e-5)


@torch.no_grad()
def test_mla_default_mask_is_causal(device):
    torch.manual_seed(0)
    batch, seq_len = 2, 10
    layer = _make_layer().to(device)
    hidden = torch.randn(batch, seq_len, 32, device=device)
    cos, sin = build_rope_cache_batched(batch, seq_len, 4, device=device)

    default_output, _ = layer(hidden, (cos, sin))
    explicit_output, _ = layer(
        hidden, (cos, sin), _causal_bool(batch, seq_len, device)
    )

    assert torch.allclose(default_output, explicit_output, atol=1e-6, rtol=1e-5)


@torch.no_grad()
def test_mla_bool_and_float_masks_are_equivalent(device):
    torch.manual_seed(0)
    batch, seq_len = 2, 10
    layer = _make_layer().to(device)
    hidden = torch.randn(batch, seq_len, 32, device=device)
    cos, sin = build_rope_cache_batched(batch, seq_len, 4, device=device)

    float_output, _ = layer(
        hidden, (cos, sin), _causal_float(batch, seq_len, device)
    )
    bool_output, _ = layer(
        hidden, (cos, sin), _causal_bool(batch, seq_len, device)
    )

    assert torch.allclose(float_output, bool_output, atol=1e-6, rtol=1e-5)


@torch.no_grad()
def test_mla_attention_does_not_leak_future_tokens(device):
    torch.manual_seed(0)
    batch, seq_len = 2, 10
    layer = _make_layer().to(device)
    hidden = torch.randn(batch, seq_len, 32, device=device)
    cos, sin = build_rope_cache_batched(batch, seq_len, 4, device=device)
    changed = hidden.clone()
    changed[:, -1] += 100.0

    output, _ = layer(hidden, (cos, sin))
    changed_output, _ = layer(changed, (cos, sin))

    assert torch.allclose(output[:, :-1], changed_output[:, :-1], atol=1e-6)
    assert not torch.allclose(output[:, -1], changed_output[:, -1])


@torch.no_grad()
def test_mla_attention_weights_sum_to_one(device):
    torch.manual_seed(0)
    batch, seq_len = 2, 9
    layer = _make_layer().to(device)
    hidden = torch.randn(batch, seq_len, 32, device=device)
    cos, sin = build_rope_cache_batched(batch, seq_len, 4, device=device)

    _, weights = layer(hidden, (cos, sin))

    assert torch.allclose(
        weights.sum(dim=-1),
        torch.ones_like(weights.sum(dim=-1)),
        atol=1e-6,
    )


@torch.no_grad()
def test_mla_accepts_longer_or_unbatched_position_embeddings(device):
    torch.manual_seed(0)
    batch, seq_len = 2, 8
    layer = _make_layer().to(device)
    hidden = torch.randn(batch, seq_len, 32, device=device)
    cos, sin = build_rope_cache_batched(batch, seq_len + 3, 4, device=device)

    long_output, _ = layer(hidden, (cos, sin))
    sliced_output, _ = layer(hidden, (cos[:, -seq_len:], sin[:, -seq_len:]))
    unbatched_output, _ = layer(hidden, (cos[0, -seq_len:], sin[0, -seq_len:]))

    assert torch.allclose(long_output, sliced_output, atol=1e-6, rtol=1e-5)
    assert torch.allclose(long_output, unbatched_output, atol=1e-6, rtol=1e-5)


def test_mla_rejects_odd_rope_dim():
    with pytest.raises(ValueError, match="even"):
        DeepseekV2Attention(
            hidden_size=32,
            num_attention_heads=4,
            kv_lora_rank=8,
            qk_nope_head_dim=8,
            qk_rope_head_dim=3,
            v_head_dim=8,
        )


@torch.no_grad()
def test_mla_rejects_bad_position_embeddings(device):
    torch.manual_seed(0)
    batch, seq_len = 1, 6
    layer = _make_layer().to(device)
    hidden = torch.randn(batch, seq_len, 32, device=device)

    wrong_dim_cos, wrong_dim_sin = build_rope_cache_batched(
        batch, seq_len, 8, device=device
    )
    with pytest.raises(ValueError, match="rotary_dim"):
        layer(hidden, (wrong_dim_cos, wrong_dim_sin))

    short_cos, short_sin = build_rope_cache_batched(
        batch, seq_len - 1, 4, device=device
    )
    with pytest.raises(ValueError, match="cover"):
        layer(hidden, (short_cos, short_sin))

    cos, sin = build_rope_cache_batched(batch, seq_len, 4, device=device)
    with pytest.raises(ValueError, match="cos, sin"):
        layer(hidden, (cos,))


def test_mla_rejects_unknown_rope_layout():
    with pytest.raises(ValueError, match="rope_layout"):
        _make_layer(rope_layout="rotary")
