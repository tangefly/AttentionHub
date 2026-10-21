# Contributing

## Core idea

The repository is organized by **algorithm**, and each backend is a file inside
that algorithm's directory. Do not add top-level `triton/` or `cuda/`
directories; a reader looking for "how is GQA implemented" should find
everything under `attention/gqa/`.

```
attention/<algorithm>/
    __init__.py     # public entry, dispatcher, available_backends()
    reference.py    # readable oracle (required for functional kernels)
    torch.py        # pure PyTorch / built-in operator
    triton.py       # Triton kernel
    cuda/           # CUDA extension: binding.cpp, <algorithm>.cu, loader
    flash.py        # flash-attn integration
    layer.py        # nn.Module variant (PSA, QSA)
```

A backend file exports the **same function name** as the algorithm:

```python
# attention/gqa/reference.py
def gqa(...): ...

# attention/gqa/triton.py
def gqa(...): ...
```

Backend selection is done by import path or by `backend=`:

```python
from attention.gqa import gqa
gqa(q, k, v, backend="auto")
```

## Naming

| Kind | Convention | Example |
| --- | --- | --- |
| Algorithm package | lowercase, canonical short name | `sdpa/`, `gqa/`, `moba/` |
| Reference implementation | `reference.py` | `attention/gqa/reference.py` |
| Pure PyTorch | `torch.py` | `attention/gqa/torch.py` |
| Triton | `triton.py` or `triton/` | `attention/moba/triton.py` |
| Custom CUDA | `cuda/` + `binding.cpp` + `<algorithm>.cu` | `attention/sdpa/cuda/sdpa.cu` |
| Third-party flash-attn | `flash.py` | `attention/moba/flash.py` |
| Module-level layer | `layer.py` | `attention/qsa/layer.py` |
| Tests | `tests/test_<algorithm>.py` | `tests/test_gqa.py` |
| Benchmarks | `benchmarks/bench_<algorithm>.py` | `benchmarks/bench_sdpa.py` |
| Docs | `docs/<algorithm>.md` | `docs/gqa.md` |
| Examples | `examples/<algorithm>.py` | `examples/gqa.py` |

Avoid `impl.py`, `utils.py`, `common.py`, `v2.py`, `new.py` and
`*_example.py`. Private helpers use a leading underscore.

## Checklist for a new kernel

1. **Reference implementation** in `attention/<algorithm>/reference.py`.
   Readable beats fast; it is the oracle for every other backend.
2. **Public entry** in `attention/<algorithm>/__init__.py` with
   `available_backends()` and `backend="auto"` dispatch.
3. **Tests** in `tests/test_<algorithm>.py`:
   - fast backends match the reference;
   - non-power-of-two sequence lengths;
   - mask/causal combinations;
   - dtype coverage (fp32 on CPU, fp16/bf16 on GPU);
   - optional backends skip when unavailable.
4. **Benchmark** in `benchmarks/bench_<algorithm>.py`, using
   `benchmarks/harness.py` (CUDA events, warmup, long-format CSV).
5. **Docs** in `docs/<algorithm>.md` with: theory, complexity, API, backend
   status, pros/cons, references.
6. **Example** in `examples/<algorithm>.py` with a `main()` and
   `if __name__ == "__main__":` guard.
7. **Registry** entry in `attention/registry.py` and a README table row
   (or regenerate the table).
8. **Optional dependency**: never import a fast backend at package import time.
   Use lazy imports / `importlib.util.find_spec`.

## Mask and layout conventions

- Boolean masks use `True` for keep; float masks are additive.
- Dense layout is `(B, H, N, D)`; packed variable-length is `(T, H, D)` plus
  `cu_seqlens`.
- `causal=True` is independent of an explicit mask.
- Public functions return `out` only by default; `return_attn=True` returns
  `(out, attention_weights)` and may be unsupported by fused kernels.

See [docs/conventions.md](docs/conventions.md) for details.

## Local commands

```bash
pip install -e ".[dev]"

# CPU tests
pytest -m "not gpu and not flash and not triton"

# Lint
ruff check .

# Examples
python examples/sdpa.py

# Benchmarks
python -m benchmarks.bench_sdpa --device cpu --seq-lens 128,256
python -m benchmarks.bench_sdpa --csv benchmarks/results/local/sdpa.csv
```

## Adding a CUDA backend

Use `torch.utils.cpp_extension.load` inside `attention/<algorithm>/cuda/__init__.py`
so the extension is built on first use and cached by PyTorch. Keep
`binding.cpp`, the `.cu` kernel and the Python loader in that directory. Add
`**/*.cu`, `**/*.cuh`, `**/*.cpp` to `[tool.setuptools.package-data]` if the
sources should ship in wheels.

Guard availability checks so that a missing compiler or GPU produces a clear
skip/`ValueError`, not an import-time crash.
