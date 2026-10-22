from __future__ import annotations

import pytest
import torch

from attention.dsa import DeepseekSparseAttention
from attention.mla import MultiHeadLatentAttention
from attention.rope import apply_rotary_pos_emb, build_rope_cache_batched


def test_dsa_is_not_an_mla_subclass():
    assert not issubclass(DeepseekSparseAttention, MultiHeadLatentAttention)


def _make_layer(
    index_topk: int = 8,
    q_lora_rank: int | None = 16,
    rope_layout: str = "interleaved",
) -> DeepseekSparseAttention:
    return DeepseekSparseAttention(
        hidden_size=32,
        num_attention_heads=4,
        kv_lora_rank=8,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        q_lora_rank=q_lora_rank,
        index_n_heads=2,
        index_head_dim=16,
        index_rope_head_dim=4,
        index_topk=index_topk,
        attention_dropout=0.0,
        attention_bias=False,
        rms_norm_eps=1e-6,
        rope_layout=rope_layout,
    ).eval()


def _make_mla(
    q_lora_rank: int | None = 16,
    rope_layout: str = "interleaved",
) -> MultiHeadLatentAttention:
    return MultiHeadLatentAttention(
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


def _copy_mla_weights(
    dsa: DeepseekSparseAttention, mla: MultiHeadLatentAttention
) -> None:
    state = {k: v for k, v in dsa.state_dict().items() if not k.startswith("indexer.")}
    mla.load_state_dict(state, strict=True)


def _causal_bool(batch: int, seq_len: int, device) -> torch.Tensor:
    causal = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).tril()
    return causal.view(1, 1, seq_len, seq_len).expand(batch, 1, seq_len, seq_len)


def _causal_float(batch: int, seq_len: int, device) -> torch.Tensor:
    mask = _causal_bool(batch, seq_len, device)
    return torch.zeros(
        batch, 1, seq_len, seq_len, device=device, dtype=torch.float32
    ).masked_fill(~mask, torch.finfo(torch.float32).min)


def _q_resid(layer: DeepseekSparseAttention, hidden_states: torch.Tensor) -> torch.Tensor:
    if layer.q_lora_rank is None:
        return hidden_states
    return layer.q_a_layernorm(layer.q_a_proj(hidden_states))


def _manual_index_scores(
    layer: DeepseekSparseAttention,
    hidden_states: torch.Tensor,
    q_resid: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Independent oracle for the indexer scoring formula."""
    indexer = layer.indexer
    batch, seq_len, _ = hidden_states.shape

    q = indexer.wq_b(q_resid).view(
        batch, seq_len, indexer.n_heads, indexer.head_dim
    )
    q_rot, q_pass = torch.split(
        q, [indexer.rope_head_dim, indexer.head_dim - indexer.rope_head_dim], dim=-1
    )
    k = indexer.k_norm(indexer.wk(hidden_states)).unsqueeze(2)
    k_rot, k_pass = torch.split(
        k, [indexer.rope_head_dim, indexer.head_dim - indexer.rope_head_dim], dim=-1
    )
    q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin, unsqueeze_dim=2)
    q = torch.cat([q_rot, q_pass], dim=-1)
    k = torch.cat([k_rot, k_pass], dim=-1).squeeze(2)

    scores = torch.matmul(
        q.float(), k.transpose(-1, -2).float().unsqueeze(1)
    ) * indexer.softmax_scale
    scores = torch.relu(scores)
    weights = indexer.weights_proj(hidden_states).float() * indexer.n_heads**-0.5
    return torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)


@pytest.mark.parametrize("rope_layout", ["interleaved", "half"])
@torch.no_grad()
def test_dsa_full_budget_matches_dense_mla(device, rope_layout):
    torch.manual_seed(0)
    batch, seq_len = 2, 12
    dsa = _make_layer(index_topk=seq_len, rope_layout=rope_layout).to(device)
    mla = _make_mla(rope_layout=rope_layout).to(device)
    _copy_mla_weights(dsa, mla)

    hidden = torch.randn(batch, seq_len, 32, device=device)
    cos, sin = build_rope_cache_batched(batch, seq_len, 4, device=device)

    output, weights = dsa(hidden, (cos, sin))
    expected_output, expected_weights = mla(hidden, (cos, sin))

    assert output.shape == (batch, seq_len, 32)
    assert weights.shape == (batch, 4, seq_len, seq_len)
    assert torch.allclose(output, expected_output, atol=1e-6, rtol=1e-5)
    assert torch.allclose(weights, expected_weights, atol=1e-6, rtol=1e-5)


@torch.no_grad()
def test_dsa_indexer_matches_manual_formula(device):
    torch.manual_seed(0)
    batch, seq_len, topk = 2, 12, 4
    layer = _make_layer(index_topk=topk).to(device)
    hidden = torch.randn(batch, seq_len, 32, device=device)
    cos, sin = build_rope_cache_batched(batch, seq_len, 4, device=device)
    mask = _causal_float(batch, seq_len, device)[:, 0]

    indices = layer.indexer(hidden, _q_resid(layer, hidden), (cos, sin), mask)
    scores = _manual_index_scores(
        layer, hidden, _q_resid(layer, hidden), cos, sin
    )
    scores = scores + mask
    expected = scores.topk(topk, dim=-1).indices

    assert indices.dtype == torch.int64
    assert indices.shape == (batch, seq_len, topk)
    assert torch.equal(indices, expected)


@torch.no_grad()
def test_dsa_matches_mla_with_selected_mask(device):
    torch.manual_seed(0)
    batch, seq_len, topk = 2, 12, 4
    dsa = _make_layer(index_topk=topk).to(device)
    mla = _make_mla().to(device)
    _copy_mla_weights(dsa, mla)
    hidden = torch.randn(batch, seq_len, 32, device=device)
    cos, sin = build_rope_cache_batched(batch, seq_len, 4, device=device)
    causal = _causal_bool(batch, seq_len, device)

    indices = dsa.indexer(
        hidden, _q_resid(dsa, hidden), (cos, sin), causal[:, 0]
    )
    keep = torch.zeros(
        batch, 1, seq_len, seq_len, dtype=torch.bool, device=device
    ).scatter(-1, indices.unsqueeze(1), True)

    output, weights = dsa(hidden, (cos, sin), causal)
    expected_output, expected_weights = mla(hidden, (cos, sin), causal & keep)

    assert torch.allclose(output, expected_output, atol=1e-6, rtol=1e-5)
    assert torch.allclose(weights, expected_weights, atol=1e-6, rtol=1e-5)


@torch.no_grad()
def test_dsa_does_not_attend_masked_keys(device):
    torch.manual_seed(0)
    batch, seq_len = 2, 12
    layer = _make_layer(index_topk=4).to(device)
    hidden = torch.randn(batch, seq_len, 32, device=device)
    cos, sin = build_rope_cache_batched(batch, seq_len, 4, device=device)

    # Mask the last three keys for every query on top of causality.
    mask = _causal_bool(batch, seq_len, device)
    mask = mask & torch.arange(seq_len, device=device).view(1, 1, 1, -1).lt(seq_len - 3)

    _, weights = layer(hidden, (cos, sin), mask)

    assert torch.all(weights[..., seq_len - 3 :] == 0)
    assert torch.allclose(
        weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-6
    )


@torch.no_grad()
def test_dsa_bool_and_float_masks_are_equivalent(device):
    torch.manual_seed(0)
    batch, seq_len = 2, 10
    layer = _make_layer(index_topk=4).to(device)
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
def test_dsa_attention_does_not_leak_future_tokens(device):
    torch.manual_seed(0)
    batch, seq_len = 2, 10
    layer = _make_layer(index_topk=3).to(device)
    hidden = torch.randn(batch, seq_len, 32, device=device)
    cos, sin = build_rope_cache_batched(batch, seq_len, 4, device=device)
    changed = hidden.clone()
    changed[:, -1] += 100.0

    output, _ = layer(hidden, (cos, sin))
    changed_output, _ = layer(changed, (cos, sin))

    assert torch.allclose(output[:, :-1], changed_output[:, :-1], atol=1e-6)
    assert not torch.allclose(output[:, -1], changed_output[:, -1])


@torch.no_grad()
def test_dsa_accepts_longer_position_embeddings(device):
    torch.manual_seed(0)
    batch, seq_len = 2, 8
    layer = _make_layer(index_topk=3).to(device)
    hidden = torch.randn(batch, seq_len, 32, device=device)
    cos, sin = build_rope_cache_batched(batch, seq_len + 3, 4, device=device)

    long_output, _ = layer(hidden, (cos, sin))
    sliced_output, _ = layer(hidden, (cos[:, -seq_len:], sin[:, -seq_len:]))

    assert torch.allclose(long_output, sliced_output, atol=1e-6, rtol=1e-5)


@torch.no_grad()
def test_dsa_without_query_lora(device):
    torch.manual_seed(0)
    batch, seq_len = 2, 9
    dsa = _make_layer(index_topk=3, q_lora_rank=None).to(device)
    mla = _make_mla(q_lora_rank=None).to(device)
    _copy_mla_weights(dsa, mla)
    assert dsa.indexer.query_dim == 32

    hidden = torch.randn(batch, seq_len, 32, device=device)
    cos, sin = build_rope_cache_batched(batch, seq_len, 4, device=device)

    causal = _causal_bool(batch, seq_len, device)
    indices = dsa.indexer(
        hidden, _q_resid(dsa, hidden), (cos, sin), causal[:, 0]
    )
    keep = torch.zeros(
        batch, 1, seq_len, seq_len, dtype=torch.bool, device=device
    ).scatter(-1, indices.unsqueeze(1), True)

    output, _ = dsa(hidden, (cos, sin))
    expected_output, _ = mla(hidden, (cos, sin), causal & keep)

    assert output.shape == (batch, seq_len, 32)
    assert torch.allclose(output, expected_output, atol=1e-6, rtol=1e-5)


@torch.no_grad()
def test_dsa_clamps_topk_to_sequence_length(device):
    torch.manual_seed(0)
    batch, seq_len = 2, 10
    layer = _make_layer(index_topk=seq_len + 8).to(device)
    hidden = torch.randn(batch, seq_len, 32, device=device)
    cos, sin = build_rope_cache_batched(batch, seq_len, 4, device=device)

    indices = layer.indexer(hidden, _q_resid(layer, hidden), (cos, sin))
    assert indices.shape == (batch, seq_len, seq_len)
    assert torch.equal(
        indices.sort(dim=-1).values,
        torch.arange(seq_len, device=device).expand(batch, seq_len, -1),
    )


def test_dsa_rejects_invalid_indexer_config():
    with pytest.raises(ValueError, match="index_topk"):
        _make_layer(index_topk=0)
    with pytest.raises(ValueError, match="index_rope_head_dim"):
        DeepseekSparseAttention(
            hidden_size=32,
            num_attention_heads=4,
            kv_lora_rank=8,
            qk_nope_head_dim=8,
            qk_rope_head_dim=4,
            v_head_dim=8,
            q_lora_rank=16,
            index_n_heads=2,
            index_head_dim=16,
            index_rope_head_dim=3,
            index_topk=4,
        )
    with pytest.raises(ValueError, match="between 0 and index_head_dim"):
        DeepseekSparseAttention(
            hidden_size=32,
            num_attention_heads=4,
            kv_lora_rank=8,
            qk_nope_head_dim=8,
            qk_rope_head_dim=4,
            v_head_dim=8,
            q_lora_rank=16,
            index_n_heads=2,
            index_head_dim=16,
            index_rope_head_dim=32,
            index_topk=4,
        )
