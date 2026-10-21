"""Shared benchmark harness.

Benchmarks are intentionally kept out of the installable package.  Run them as
modules from the repository root, e.g.::

    python -m benchmarks.bench_sdpa --device cpu --seq-lens 128,256
"""

from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass
class BenchmarkResult:
    name: str
    backend: str
    device: str
    dtype: str
    batch: int
    heads: int
    seq_len: int
    head_dim: int
    time_ms: float
    tflops: float


def attention_flops(batch: int, heads: int, seq_len: int, head_dim: int) -> int:
    """Dense-equivalent FLOPs: QK^T and AV."""
    return 4 * batch * heads * seq_len * seq_len * head_dim


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def benchmark(
    name: str,
    fn,
    *,
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    backend: str = "auto",
    warmup: int = 5,
    iters: int = 20,
) -> BenchmarkResult:
    """Time ``fn`` with device-appropriate synchronization."""
    for _ in range(warmup):
        fn()
    _synchronize(device)

    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        time_ms = start.elapsed_time(end) / iters
    else:
        start_time = time.perf_counter()
        for _ in range(iters):
            fn()
        time_ms = (time.perf_counter() - start_time) * 1000 / iters

    flops = attention_flops(batch, heads, seq_len, head_dim)
    tflops = flops / (time_ms / 1000) / 1e12
    return BenchmarkResult(
        name=name,
        backend=backend,
        device=str(device),
        dtype=str(dtype).replace("torch.", ""),
        batch=batch,
        heads=heads,
        seq_len=seq_len,
        head_dim=head_dim,
        time_ms=time_ms,
        tflops=tflops,
    )


def save_csv(results: list[BenchmarkResult], path: str | Path) -> None:
    """Write results in long format so they can be concatenated across GPUs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def format_result(result: BenchmarkResult) -> str:
    return (
        f"{result.name:>12} backend={result.backend:<10} "
        f"seq={result.seq_len:<6} {result.time_ms:8.3f} ms  "
        f"{result.tflops:7.2f} TFLOPS"
    )
