from __future__ import annotations

import torch

from attention.psa import PSA, PredictSparseAttention


def test_psa_output_shape(device, dtype):
    torch.manual_seed(0)
    x = torch.randn(2, 16, 32, device=device, dtype=dtype)
    layer = PredictSparseAttention(d=32, k=8, topk=4).to(device)

    output = layer(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_psa_alias():
    assert PSA is PredictSparseAttention
