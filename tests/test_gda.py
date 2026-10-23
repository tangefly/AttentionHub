from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from attention.gda import (
    GatedDeltaNet,
    GatedDeltaNetState,
    GatedRMSNorm,
    Qwen3_5GatedDeltaNet,
    gated_delta_rule,
    recurrent_gated_delta_rule,
)
from attention.gda import reference as gda_reference
from attention.gda import torch as gda_torch
from tests.utils import tolerance


def _inputs(
    device: torch.device,
    batch: int = 2,
    seq_len: int = 37,
    heads: int = 3,
    head_k_dim: int = 8,
    head_v_dim: int = 6,
):
    torch.manual_seed(0)
    q = torch.randn(batch, seq_len, heads, head_k_dim, device=device)
    k = torch.randn(batch, seq_len, heads, head_k_dim, device=device)
    v = torch.randn(batch, seq_len, heads, head_v_dim, device=device)
    g = -torch.rand(batch, seq_len, heads, device=device)
    beta = torch.rand(batch, seq_len, heads, device=device)
    return q, k, v, g, beta


def _make_layer(
    hidden_size: int = 32,
    num_v_heads: int = 6,
    num_k_heads: int = 3,
    head_k_dim: int = 8,
    head_v_dim: int = 6,
    conv_kernel_size: int = 4,
) -> GatedDeltaNet:
    return GatedDeltaNet(
        hidden_size=hidden_size,
        num_v_heads=num_v_heads,
        num_k_heads=num_k_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        conv_kernel_size=conv_kernel_size,
        rms_norm_eps=1e-6,
        activation="silu",
    ).eval()


@pytest.mark.parametrize("seq_len", [1, 5, 37, 64, 65])
@pytest.mark.parametrize("chunk_size", [1, 7, 16, 64])
def test_chunked_matches_reference(device, seq_len, chunk_size):
    q, k, v, g, beta = _inputs(device, seq_len=seq_len)
    out_ref, _ = gda_reference.gated_delta_rule(q, k, v, g, beta)
    out_chunk, _ = gda_torch.gated_delta_rule(
        q, k, v, g, beta, chunk_size=chunk_size
    )
    assert out_chunk.shape == out_ref.shape == (2, seq_len, 3, 6)
    assert torch.allclose(
        out_chunk, out_ref, atol=tolerance(torch.float32), rtol=1e-4
    )


def test_recurrent_matches_reference_and_chunked(device):
    q, k, v, g, beta = _inputs(device)
    out_ref, state_ref = gda_reference.gated_delta_rule(
        q, k, v, g, beta, output_final_state=True
    )
    out_rec, state_rec = recurrent_gated_delta_rule(
        q, k, v, g, beta, output_final_state=True
    )
    # The recurrent kernel mirrors ``transformers``' reduction order
    # (``(state * x).sum(-2)`` instead of ``einsum``), so it agrees with the
    # reference up to float32 rounding rather than bit for bit.
    assert torch.allclose(
        out_rec, out_ref, atol=tolerance(torch.float32), rtol=1e-4
    )
    assert torch.allclose(
        state_rec, state_ref, atol=tolerance(torch.float32), rtol=1e-4
    )


def test_initial_and_final_state(device):
    q, k, v, g, beta = _inputs(device)
    initial_state = torch.randn(2, 3, 8, 6, device=device)

    out_ref, state_ref = gda_reference.gated_delta_rule(
        q, k, v, g, beta, initial_state=initial_state, output_final_state=True
    )
    out_chunk, state_chunk = gda_torch.gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        chunk_size=16,
        initial_state=initial_state,
        output_final_state=True,
    )
    out_rec, state_rec = recurrent_gated_delta_rule(
        q, k, v, g, beta, initial_state=initial_state, output_final_state=True
    )

    atol = tolerance(torch.float32)
    assert torch.allclose(out_chunk, out_ref, atol=atol, rtol=1e-4)
    assert torch.allclose(out_rec, out_ref, atol=atol, rtol=1e-4)
    assert torch.allclose(state_chunk, state_ref, atol=atol, rtol=1e-4)
    assert torch.allclose(state_rec, state_ref, atol=atol, rtol=1e-4)


def test_l2norm_and_scale_variants(device):
    q, k, v, g, beta = _inputs(device)
    out_ref, _ = gda_reference.gated_delta_rule(
        q, k, v, g, beta, use_qk_l2norm_in_kernel=True, scale=0.5
    )
    out_chunk, _ = gda_torch.gated_delta_rule(
        q, k, v, g, beta, use_qk_l2norm_in_kernel=True, scale=0.5, chunk_size=16
    )
    assert torch.allclose(
        out_chunk, out_ref, atol=tolerance(torch.float32), rtol=1e-4
    )


def test_backend_wrapper(device):
    q, k, v, g, beta = _inputs(device)
    out_ref, _ = gated_delta_rule(q, k, v, g, beta, backend="reference")
    out_torch, _ = gated_delta_rule(q, k, v, g, beta, backend="torch")
    assert torch.allclose(out_ref, out_torch, atol=tolerance(torch.float32))
    with pytest.raises(ValueError, match="not available"):
        gated_delta_rule(q, k, v, g, beta, backend="cuda")


def test_layer_matches_manual_composition(device):
    torch.manual_seed(0)
    layer = _make_layer().to(device)
    x = torch.randn(2, 17, 32, device=device)

    with torch.no_grad():
        output = layer(x)

        batch, seq_len, _ = x.shape
        mixed = layer.in_proj_qkv(x).transpose(1, 2)
        conv = F.conv1d(
            mixed,
            layer.conv1d.weight,
            layer.conv1d.bias,
            padding=layer.conv_kernel_size - 1,
            groups=layer.conv_dim,
        )[..., :seq_len]
        conv = F.silu(conv).transpose(1, 2)
        q, k, v = torch.split(
            conv, [layer.key_dim, layer.key_dim, layer.value_dim], dim=-1
        )
        q = q.view(batch, seq_len, layer.num_k_heads, layer.head_k_dim)
        k = k.view(batch, seq_len, layer.num_k_heads, layer.head_k_dim)
        v = v.view(batch, seq_len, layer.num_v_heads, layer.head_v_dim)
        z = layer.in_proj_z(x).view(
            batch, seq_len, layer.num_v_heads, layer.head_v_dim
        )
        beta = layer.in_proj_b(x).sigmoid()
        g = -layer.A_log.float().exp() * F.softplus(
            layer.in_proj_a(x).float() + layer.dt_bias
        )
        if layer.num_key_value_groups > 1:
            q = q.repeat_interleave(layer.num_key_value_groups, dim=2)
            k = k.repeat_interleave(layer.num_key_value_groups, dim=2)

        core, _ = gda_reference.gated_delta_rule(
            q, k, v, g, beta, use_qk_l2norm_in_kernel=layer.use_qk_l2norm
        )
        core = core.reshape(-1, layer.head_v_dim)
        z = z.reshape(-1, layer.head_v_dim)
        normed = core.float()
        normed = normed * torch.rsqrt(
            normed.pow(2).mean(-1, keepdim=True) + layer.norm.variance_epsilon
        )
        normed = layer.norm.weight * normed.to(core.dtype)
        normed = (normed * F.silu(z.float())).to(core.dtype)
        expected = layer.out_proj(
            normed.view(batch, seq_len, layer.value_dim)
        )

    assert torch.allclose(
        output, expected, atol=tolerance(torch.float32), rtol=1e-4
    )


def test_incremental_decoding_matches_full_forward(device):
    torch.manual_seed(0)
    layer = _make_layer().to(device)
    x = torch.randn(2, 19, 32, device=device)

    with torch.no_grad():
        full, full_state = layer(x, use_cache=True)
        assert isinstance(full_state, GatedDeltaNetState)
        assert full_state.recurrent_state.dtype == torch.float32
        assert full_state.recurrent_state.shape == (2, 6, 8, 6)
        assert full_state.conv_state.shape == (2, layer.conv_dim, 3)

        # Decode from scratch.
        steps = []
        state = None
        for t in range(x.shape[1]):
            step, state = layer(x[:, t : t + 1], past_state=state, use_cache=True)
            steps.append(step)
        decoded = torch.cat(steps, dim=1)
        assert torch.allclose(
            decoded, full, atol=tolerance(torch.float32), rtol=1e-4
        )

        # Prefill a chunk, then decode the rest.
        prefill, state = layer(x[:, :7], use_cache=True)
        chunks = [prefill]
        for t in range(7, x.shape[1]):
            step, state = layer(x[:, t : t + 1], past_state=state, use_cache=True)
            chunks.append(step)
        cached = torch.cat(chunks, dim=1)
        assert torch.allclose(
            cached, full, atol=tolerance(torch.float32), rtol=1e-4
        )

        # Multi-token continuation: prefill + one chunked forward carrying state.
        prefill, state = layer(x[:, :7], use_cache=True)
        rest, state = layer(x[:, 7:], past_state=state, use_cache=True)
        continued = torch.cat([prefill, rest], dim=1)
        assert torch.allclose(
            continued, full, atol=tolerance(torch.float32), rtol=1e-4
        )


def test_padding_mask(device):
    torch.manual_seed(0)
    layer = _make_layer().to(device)
    x = torch.randn(2, 13, 32, device=device)
    valid = 10

    bool_mask = torch.ones(2, 13, dtype=torch.bool, device=device)
    bool_mask[:, valid:] = False
    float_mask = bool_mask.to(x.dtype)
    additive_mask = torch.zeros(2, 13, device=device, dtype=x.dtype)
    additive_mask[:, valid:] = torch.finfo(x.dtype).min

    with torch.no_grad():
        out_bool = layer(x, attention_mask=bool_mask)
        out_float = layer(x, attention_mask=float_mask)
        out_additive = layer(x, attention_mask=additive_mask)
        out_prefix = layer(x[:, :valid])

    assert torch.allclose(out_bool, out_float, atol=1e-6, rtol=1e-5)
    assert torch.allclose(out_bool, out_additive, atol=1e-6, rtol=1e-5)
    # Right padding is causal: valid positions must match the unpadded prefix.
    assert torch.allclose(
        out_bool[:, :valid], out_prefix, atol=1e-6, rtol=1e-5
    )


def test_gated_rms_norm_matches_manual(device):
    torch.manual_seed(0)
    norm = GatedRMSNorm(6, eps=1e-6).to(device)
    x = torch.randn(4, 6, device=device)
    gate = torch.randn(4, 6, device=device)

    out = norm(x, gate)
    x_f = x.float()
    normed = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + 1e-6)
    expected = (norm.weight * normed.to(x.dtype)) * F.silu(gate.float())
    expected = expected.to(x.dtype)
    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6)


def test_aliases_and_parameter_names():
    assert Qwen3_5GatedDeltaNet is GatedDeltaNet
    layer = _make_layer()
    for name in (
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "conv1d",
        "dt_bias",
        "A_log",
        "norm",
        "out_proj",
    ):
        assert hasattr(layer, name), name


def test_invalid_arguments():
    with pytest.raises(ValueError, match="divisible"):
        GatedDeltaNet(32, 5, 3, 8, 6)
    with pytest.raises(ValueError, match="conv_kernel_size"):
        GatedDeltaNet(32, 6, 3, 8, 6, conv_kernel_size=0)
    with pytest.raises(ValueError, match="activation"):
        GatedDeltaNet(32, 6, 3, 8, 6, activation="nope")

    layer = _make_layer()
    with pytest.raises(ValueError, match="2D padding mask"):
        layer(torch.randn(2, 4, 32), attention_mask=torch.ones(2, 1, 4, 4))


def test_conv_kernel_one(device):
    torch.manual_seed(0)
    layer = _make_layer(conv_kernel_size=1).to(device)
    x = torch.randn(2, 9, 32, device=device)
    with torch.no_grad():
        full, state = layer(x, use_cache=True)
        assert state.conv_state.shape == (2, layer.conv_dim, 0)
        _, prefix_state = layer(x[:, :-1], use_cache=True)
        step, _ = layer(x[:, -1:], past_state=prefix_state, use_cache=True)
    assert torch.allclose(
        step, full[:, -1:], atol=tolerance(torch.float32), rtol=1e-4
    )


def _transformers_gda_kernels():
    """Return the eager reference kernels shipped with ``transformers``.

    The installed ``transformers`` dispatches these symbols to FLA's Triton
    kernels when available, so ``__wrapped__`` is used to reach the pure
    PyTorch fallback that :mod:`attention.gda.torch` is ported from.
    """
    try:
        modeling = pytest.importorskip(
            "transformers.models.qwen3_5.modeling_qwen3_5"
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"transformers' Qwen3.5 kernels are unavailable: {exc}")

    chunk = getattr(modeling, "torch_chunk_gated_delta_rule", None)
    recurrent = getattr(modeling, "torch_recurrent_gated_delta_rule", None)
    if chunk is None or recurrent is None:  # pragma: no cover - version drift
        pytest.skip("installed transformers does not expose the torch GDA kernels")
    return getattr(chunk, "__wrapped__", chunk), getattr(
        recurrent, "__wrapped__", recurrent
    )


@pytest.mark.parametrize("chunk_size", [16, 64])
def test_chunked_matches_transformers_torch_kernel(device, chunk_size):
    chunk, _ = _transformers_gda_kernels()
    q, k, v, g, beta = _inputs(device)
    initial_state = torch.randn(2, 3, 8, 6, device=device)
    kwargs = dict(
        chunk_size=chunk_size,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    out, state = gda_torch.gated_delta_rule(q, k, v, g, beta, **kwargs)
    ref_out, ref_state = chunk(q, k, v, g, beta, **kwargs)
    # The port is line-for-line, so the eager results are expected to be
    # bit-identical rather than merely close.
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
    torch.testing.assert_close(state, ref_state, rtol=0, atol=0)


def test_recurrent_matches_transformers_torch_kernel(device):
    _, recurrent = _transformers_gda_kernels()
    q, k, v, g, beta = _inputs(device)
    initial_state = torch.randn(2, 3, 8, 6, device=device)
    kwargs = dict(
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    out, state = recurrent_gated_delta_rule(q, k, v, g, beta, **kwargs)
    ref_out, ref_state = recurrent(q, k, v, g, beta, **kwargs)
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
    torch.testing.assert_close(state, ref_state, rtol=0, atol=0)


def test_export_fallback_matches_triangular_solve(device, monkeypatch):
    q, k, v, g, beta = _inputs(device)
    initial_state = torch.randn(2, 3, 8, 6, device=device)
    kwargs = dict(
        chunk_size=16,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    fast_out, fast_state = gda_torch.gated_delta_rule(q, k, v, g, beta, **kwargs)
    monkeypatch.setattr(gda_torch, "_is_torchdynamo_exporting", lambda: True)
    slow_out, slow_state = gda_torch.gated_delta_rule(q, k, v, g, beta, **kwargs)
    atol = tolerance(torch.float32)
    assert torch.allclose(fast_out, slow_out, atol=atol, rtol=1e-5)
    assert torch.allclose(fast_state, slow_state, atol=atol, rtol=1e-5)


def test_accepts_transformers_kernel_kwargs(device):
    q, k, v, g, beta = _inputs(device)
    # ``transformers`` call sites pass kernel-only kwargs such as cu_seqlens.
    out_chunk, _ = gda_torch.gated_delta_rule(
        q, k, v, g, beta, chunk_size=16, cu_seqlens=None
    )
    out_rec, _ = gda_torch.recurrent_gated_delta_rule(
        q, k, v, g, beta, cu_seqlens=None
    )
    assert out_chunk.shape == out_rec.shape == (2, 37, 3, 6)
