# AttentionHub

Collection of attention variants and their implementations: readable reference
code, pure PyTorch, flash-attn, Triton and custom CUDA kernels.

The project is organized **by algorithm**, not by backend. Each algorithm
directory contains one file per backend, and every backend exposes the same
public function:

```
attention/<algorithm>/
    reference.py   # readable oracle, always present
    torch.py       # pure PyTorch / built-in operator
    triton.py      # Triton kernel
    cuda/          # custom CUDA extension
    flash.py       # flash-attn integration
    layer.py       # nn.Module variant (PSA, QSA)
```

## Install

```bash
git clone https://github.com/tangefly/AttentionHub.git
cd AttentionHub
pip install -e .
```

Optional backends are installed separately:

```bash
pip install -e ".[triton]"   # Triton kernels
pip install -e ".[flash]"    # flash-attn (requires CUDA)
pip install -e ".[dev]"      # pytest + ruff
```

Requires Python 3.10+.

## Quick start

```python
import torch
from attention.sdpa import sdpa

q = torch.randn(2, 8, 1024, 64)
k = torch.randn(2, 8, 1024, 64)
v = torch.randn(2, 8, 1024, 64)

# Pick the fastest installed backend automatically.
out = sdpa(q, k, v, causal=True)

# Reference implementation, with attention weights for inspection.
out_ref, weights = sdpa(
    q, k, v, causal=True, backend="reference", return_attn=True
)
```

Other kernels follow the same pattern:

```python
from attention.gqa import gqa
from attention.moba import moba
from attention.qsa import QwenSparseAttention
```

`available_backends()` reports what can run in the current environment:

```python
import attention
attention.available_backends("sdpa")  # ('reference', 'torch')
```

## RoPE utilities

RoPE cache construction and application live in
[attention/rope.py](attention/rope.py):

```python
from attention.rope import apply_rotary_pos_emb, build_rope_cache_batched

cos, sin = build_rope_cache_batched(
    batch, seq_len, rotary_dim, device=device, dtype=q.dtype
)
q, k = apply_rotary_pos_emb(q, k, cos, sin)
```

See [docs/rope.md](docs/rope.md) for the full API.

## Supported kernels

| Year | Kernel | Docs | Paper | Reference | Torch | Flash | Triton | CUDA |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 2017 | Scaled Dot-Product Attention (SDPA) | [SDPA](docs/sdpa.md) | [Link](https://arxiv.org/abs/1706.03762) | ✅ | ✅ | ❌ | ❌ | ❌ |
| 2019 | Factorized Attention (FTA) | [FTA](docs/fta.md) | [Link](https://arxiv.org/abs/1904.10509) | ✅ | ❌ | ❌ | ❌ | ❌ |
| 2019 | Multi-Query Attention (MQA) | [MQA](docs/mqa.md) | [Link](https://arxiv.org/abs/1911.02150) | ✅ | ❌ | ❌ | ❌ | ❌ |
| 2022 | Predictive Sparse Attention (PSA) | [PSA](docs/psa.md) | [Link](https://arxiv.org/abs/2110.11299) | ✅ | ❌ | ❌ | ❌ | ❌ |
| 2023 | Grouped-Query Attention (GQA) | [GQA](docs/gqa.md) | [Link](https://arxiv.org/abs/2305.13245) | ✅ | ❌ | ❌ | ❌ | ❌ |
| 2025 | Mixture of Block Attention (MoBA) | [MoBA](docs/moba.md) | [Link](https://arxiv.org/abs/2502.13189) | ✅ | ❌ | ✅ | ❌ | ❌ |
| 2026 | Qwen Sparse Attention (QSA) | [QSA](docs/qsa.md) | - | ✅ | ❌ | ❌ | ❌ | ❌ |

The matrix is generated from [`attention/registry.py`](attention/registry.py)
(`python -m attention.registry`), so documentation and code stay in sync.

## Project layout

```
attention/              # installable package
  registry.py          # kernel metadata / support matrix
  dispatch.py          # backend="auto" resolution
  mask.py              # shared mask conventions
  rope.py              # shared RoPE utilities
  sdpa/                 # one directory per algorithm
    __init__.py         # public entry + available_backends()
    reference.py
    torch.py
  ...
examples/               # runnable examples, one per algorithm
benchmarks/             # shared harness + bench_<algorithm>.py
tests/                  # cross-backend correctness tests
docs/                   # theory, API and conventions
```

Shared conventions (tensor layouts, mask semantics, return values, backend
order) are documented in [docs/conventions.md](docs/conventions.md).

## Development

Run the CPU test suite:

```bash
pytest -m "not gpu and not flash and not triton"
```

Run an example:

```bash
python examples/sdpa.py
```

Run a benchmark (results are written as long-format CSV):

```bash
python -m benchmarks.bench_sdpa --device cpu --seq-lens 128,256
python -m benchmarks.bench_sdpa --csv benchmarks/results/local/sdpa.csv
```

Flash/Triton/CUDA tests are marked and skipped automatically when the backend is
not available. See [CONTRIBUTING.md](CONTRIBUTING.md) for the checklist to add a
new kernel or backend.

## License

Apache-2.0. See [LICENSE](LICENSE).
