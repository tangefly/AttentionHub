from __future__ import annotations

import argparse

import torch

from attention.moba import available_backends, moba
from benchmarks.harness import benchmark, format_result, save_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MoBA backends")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--seq-lens", default="64,128")
    parser.add_argument("--backends", default="reference,flash")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--csv", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if args.dtype == "auto":
        dtype = torch.float16 if device.type == "cuda" else torch.float32
    else:
        dtype = getattr(torch, args.dtype)

    print(f"device={device} dtype={dtype}")
    results = []
    backends = [item.strip() for item in args.backends.split(",") if item.strip()]
    for seq_len in (int(item) for item in args.seq_lens.split(",")):
        total = args.batch * seq_len
        q = torch.randn(total, args.heads, args.head_dim, device=device, dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        cu_seqlens = torch.arange(
            0, total + 1, seq_len, device=device, dtype=torch.int32
        )
        for backend in backends:
            if backend not in available_backends():
                print(f"{backend}: not available, skipping")
                continue
            result = benchmark(
                "moba",
                lambda backend=backend: moba(
                    q,
                    k,
                    v,
                    cu_seqlens,
                    max_seqlen=seq_len,
                    moba_chunk_size=args.chunk_size,
                    moba_topk=args.topk,
                    backend=backend,
                ),
                batch=args.batch,
                heads=args.heads,
                seq_len=seq_len,
                head_dim=args.head_dim,
                device=device,
                dtype=dtype,
                backend=backend,
                warmup=args.warmup,
                iters=args.iters,
            )
            results.append(result)
            print(format_result(result))

    if args.csv and results:
        save_csv(results, args.csv)
        print(f"saved {args.csv}")


if __name__ == "__main__":
    main()
