"""Compare the pure-PyTorch gated delta rule with FLA's Triton kernels.

Both implementations compute the same recurrence (see
:mod:`attention.gda.reference`); this script only measures how fast they do it
on the shapes a Qwen3.5-style GDA layer would use.

What differs
------------
``attention.gda.torch``
    Chunked UT-transform written in eager PyTorch.  It upcasts q/k/v to float32
    (for numerical stability) and materializes the per-chunk ``C x C`` decay and
    attention matrices.  Grouped value attention is emulated with
    ``repeat_interleave`` on q/k, exactly like ``GatedDeltaNet`` does.

``fla.ops.gated_delta_rule``
    Fused Triton kernels.  They keep the recurrent state on chip, solve the
    triangular system inside the kernel, run on tensor cores in bf16/fp16, and
    support grouped value heads natively (no repeated q/k).

Because the torch backend always computes in fp32, this is not a pure
apples-to-apples kernel comparison -- it is a comparison of the two *backends
as you would actually use them*.  The timing of the torch path therefore
includes the ``repeat_interleave`` needed for GVA.

Usage
-----
::

    python -m benchmarks.bench_gda                       # CUDA/bf16, auto shapes
    python -m benchmarks.bench_gda --seq-lens 512,2048,8192
    python -m benchmarks.bench_gda --batch 1 --v-heads 4
    python -m benchmarks.bench_gda --csv benchmarks/results/local/gda.csv

On a CPU-only machine the script still runs and prints the torch baseline, but
the FLA columns stay empty (Triton needs CUDA).  CPU timings are dominated by
eager PyTorch launch overhead and thread oversubscription, so use
``--threads 1`` (or ``--threads 8``) there to get stable numbers.
"""

from __future__ import annotations

import argparse
import math

import torch

from attention.gda import torch as gda_torch
from benchmarks.harness import BenchmarkResult, benchmark, save_csv

try:  # FLA is optional and its kernels are CUDA-only.
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule as fla_chunk_gated_delta_rule,
    )
    from fla.ops.gated_delta_rule import (
        fused_recurrent_gated_delta_rule as fla_recurrent_gated_delta_rule,
    )
except Exception:  # ImportError, or a broken platform-specific build
    fla_chunk_gated_delta_rule = None
    fla_recurrent_gated_delta_rule = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark GDA: torch vs FLA")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"]
    )
    parser.add_argument("--seq-lens", default="256,512,1024,2048,4096")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--k-heads", type=int, default=4)
    parser.add_argument("--v-heads", type=int, default=8)
    parser.add_argument("--head-k-dim", type=int, default=64)
    parser.add_argument("--head-v-dim", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help=(
            "CPU only: call torch.set_num_threads(). Eager PyTorch issues many "
            "tiny kernels, so oversubscription can dominate the measurement; "
            "try --threads 1..8 for stable CPU numbers."
        ),
    )
    parser.add_argument(
        "--decode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also benchmark the T=1 (token-by-token) decoding path",
    )
    parser.add_argument("--csv", default=None)
    return parser.parse_args()


def _fla_ready(device: torch.device) -> bool:
    return fla_chunk_gated_delta_rule is not None and device.type == "cuda"


def make_inputs(
    batch: int,
    seq_len: int,
    k_heads: int,
    v_heads: int,
    k_dim: int,
    v_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random inputs in the layout both backends expect.

    ``g`` and ``beta`` stay float32, exactly like ``GatedDeltaNet`` computes
    them; the kernels cast internally.
    """
    shape_qk = (batch, seq_len, k_heads, k_dim)
    q = torch.randn(shape_qk, device=device, dtype=dtype)
    k = torch.randn(shape_qk, device=device, dtype=dtype)
    v = torch.randn(batch, seq_len, v_heads, v_dim, device=device, dtype=dtype)
    # Log decay in [-4, 0) and a post-sigmoid write strength.
    g = -torch.rand(batch, seq_len, v_heads, device=device).mul(4)
    beta = torch.rand(batch, seq_len, v_heads, device=device).sigmoid()
    return q, k, v, g, beta


def gda_flops(
    batch: int,
    seq_len: int,
    v_heads: int,
    k_dim: int,
    v_dim: int,
    chunk_size: int = 64,
) -> int:
    """Rough MAC count: 3 state passes per token plus the intra-chunk matmuls.

    Per token the state ``(K, V)`` is read for the delta, written by the rank-one
    update and read again for the output -> ``3 * K * V`` MACs.  The intra-chunk
    ``QK^T`` and ``A V`` work amortizes to ``C * (K + V)`` per token.  This
    ignores the (cheap) triangular solve and the element-wise decay, so treat it
    as an order-of-magnitude figure.
    """
    intra = chunk_size * (k_dim + v_dim) if seq_len > 1 else 0
    per_token = 3 * k_dim * v_dim + intra
    return 2 * batch * v_heads * seq_len * per_token


def _diff(got: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    delta = (got - expected).abs()
    relative = delta / expected.abs().clamp_min(1e-3)
    return delta.max().item(), relative.max().item()


def main() -> int:
    args = parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is not available on this machine.")
        return 1
    if args.dtype == "auto":
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    else:
        dtype = getattr(torch, args.dtype)

    if args.v_heads % args.k_heads != 0:
        print(
            f"v_heads ({args.v_heads}) must be divisible by "
            f"k_heads ({args.k_heads})"
        )
        return 1

    groups = args.v_heads // args.k_heads
    fla_ok = _fla_ready(device)
    if fla_ok and args.chunk_size not in (16, 32, 64):
        print("FLA's chunked kernel only supports chunk_size 16, 32 or 64.")
        return 1

    print("GDA kernel benchmark: torch (eager, chunked) vs FLA (Triton)")
    print(
        f"  device={device} dtype={dtype} batch={args.batch} "
        f"k_heads={args.k_heads} v_heads={args.v_heads} "
        f"head_k={args.head_k_dim} head_v={args.head_v_dim} "
        f"chunk={args.chunk_size}"
    )
    if fla_ok:
        print("  warmup calls absorb FLA's first-call kernel selection")
    else:
        print("\nFLA is not usable here -- showing the torch baseline only.")
        if device.type != "cuda":
            print("  reason: FLA's kernels are Triton kernels and need CUDA")
        else:
            print("  reason: fla is not installed (pip install flash-linear-attention)")
        print("  run on a GPU machine to get the comparison.")

    results: list[BenchmarkResult] = []
    speedups: list[float] = []

    if fla_ok:
        print(
            f"\n{'seq_len':>8} {'torch ms':>10} {'fla ms':>10} "
            f"{'speedup':>8} {'max|diff|':>10} {'max rel':>10}"
        )
    else:
        print(f"\n{'seq_len':>8} {'torch ms':>10}")

    for seq_len in (int(item) for item in args.seq_lens.split(",")):
        q, k, v, g, beta = make_inputs(
            args.batch,
            seq_len,
            args.k_heads,
            args.v_heads,
            args.head_k_dim,
            args.head_v_dim,
            device,
            dtype,
        )
        flops = gda_flops(
            args.batch,
            seq_len,
            args.v_heads,
            args.head_k_dim,
            args.head_v_dim,
            args.chunk_size,
        )

        def run_torch(q=q, k=k, v=v, g=g, beta=beta, groups=groups):
            # Mirror GatedDeltaNet: the torch backend needs q/k per value head.
            q_rep = q.repeat_interleave(groups, dim=2) if groups > 1 else q
            k_rep = k.repeat_interleave(groups, dim=2) if groups > 1 else k
            return gda_torch.gated_delta_rule(
                q_rep,
                k_rep,
                v,
                g,
                beta,
                chunk_size=args.chunk_size,
                use_qk_l2norm_in_kernel=True,
            )

        def run_fla(q=q, k=k, v=v, g=g, beta=beta):
            # FLA maps value head h_v to key head h_v // groups internally.
            return fla_chunk_gated_delta_rule(
                q,
                k,
                v,
                g,
                beta,
                chunk_size=args.chunk_size,
                use_qk_l2norm_in_kernel=True,
            )

        with torch.no_grad():
            out_torch = run_torch()[0].float()
            result_torch = benchmark(
                "gda",
                run_torch,
                batch=args.batch,
                heads=args.v_heads,
                seq_len=seq_len,
                head_dim=args.head_k_dim,
                device=device,
                dtype=dtype,
                backend="torch",
                warmup=args.warmup,
                iters=args.iters,
                flops=flops,
            )
            results.append(result_torch)

            if not fla_ok:
                print(f"{seq_len:>8} {result_torch.time_ms:>10.3f}")
                continue

            out_fla = run_fla()[0].float()
            result_fla = benchmark(
                "gda",
                run_fla,
                batch=args.batch,
                heads=args.v_heads,
                seq_len=seq_len,
                head_dim=args.head_k_dim,
                device=device,
                dtype=dtype,
                backend="fla",
                warmup=args.warmup,
                iters=args.iters,
                flops=flops,
            )
            results.append(result_fla)

        speedup = result_torch.time_ms / result_fla.time_ms
        speedups.append(speedup)
        max_abs, max_rel = _diff(out_fla, out_torch)
        print(
            f"{seq_len:>8} {result_torch.time_ms:>10.3f} "
            f"{result_fla.time_ms:>10.3f} {speedup:>7.2f}x "
            f"{max_abs:>10.2e} {max_rel:>10.2e}"
        )

    if speedups:
        geomean = math.exp(sum(math.log(item) for item in speedups) / len(speedups))
        print(
            f"\navg prefill speedup: {geomean:.2f}x "
            f"(geometric mean over {len(speedups)} sequence lengths)"
        )
        print(
            "note: torch computes in fp32 internally (no tensor cores, higher "
            "precision) and pays for\n      the GVA repeat; FLA runs in the input "
            "dtype and handles GVA natively."
        )

    if args.decode:
        print("\ndecode: one token per call (T=1), recurrent state carried")
        q1, k1, v1, g1, b1 = make_inputs(
            args.batch,
            1,
            args.k_heads,
            args.v_heads,
            args.head_k_dim,
            args.head_v_dim,
            device,
            dtype,
        )
        state = torch.zeros(
            args.batch,
            args.v_heads,
            args.head_k_dim,
            args.head_v_dim,
            dtype=torch.float32,
            device=device,
        )

        def run_torch_decode(q1=q1, k1=k1, v1=v1, g1=g1, b1=b1):
            q_rep = q1.repeat_interleave(groups, dim=2) if groups > 1 else q1
            k_rep = k1.repeat_interleave(groups, dim=2) if groups > 1 else k1
            return gda_torch.recurrent_gated_delta_rule(
                q_rep,
                k_rep,
                v1,
                g1,
                b1,
                initial_state=state,
                use_qk_l2norm_in_kernel=True,
            )

        def run_fla_decode(q1=q1, k1=k1, v1=v1, g1=g1, b1=b1):
            return fla_recurrent_gated_delta_rule(
                q=q1,
                k=k1,
                v=v1,
                g=g1,
                beta=b1,
                initial_state=state,
                use_qk_l2norm_in_kernel=True,
            )

        decode_flops = gda_flops(
            args.batch, 1, args.v_heads, args.head_k_dim, args.head_v_dim, 1
        )
        with torch.no_grad():
            step_torch = run_torch_decode()[0].float()
            result_torch = benchmark(
                "gda-decode",
                run_torch_decode,
                batch=args.batch,
                heads=args.v_heads,
                seq_len=1,
                head_dim=args.head_k_dim,
                device=device,
                dtype=dtype,
                backend="torch",
                warmup=args.warmup,
                iters=args.iters,
                flops=decode_flops,
            )
            results.append(result_torch)
            print(
                f"  torch {result_torch.time_ms:8.4f} ms/step "
                f"({1000 / result_torch.time_ms:8.0f} steps/s)"
            )

            if fla_ok:
                step_fla = run_fla_decode()[0].float()
                result_fla = benchmark(
                    "gda-decode",
                    run_fla_decode,
                    batch=args.batch,
                    heads=args.v_heads,
                    seq_len=1,
                    head_dim=args.head_k_dim,
                    device=device,
                    dtype=dtype,
                    backend="fla",
                    warmup=args.warmup,
                    iters=args.iters,
                    flops=decode_flops,
                )
                results.append(result_fla)
                max_abs, max_rel = _diff(step_fla, step_torch)
                print(
                    f"  fla   {result_fla.time_ms:8.4f} ms/step "
                    f"({1000 / result_fla.time_ms:8.0f} steps/s)  "
                    f"speedup {result_torch.time_ms / result_fla.time_ms:.2f}x  "
                    f"max|diff| {max_abs:.2e}"
                )

    if args.csv and results:
        save_csv(results, args.csv)
        print(f"\nsaved {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
