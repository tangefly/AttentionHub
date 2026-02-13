import matplotlib.pyplot as plt
import os
import torch
import time

def attention_flops(b, h, n, d):
    # QK^T + AV
    return 4 * b * h * n * n * d


def run_attention_benchmark(
    name,
    fn,
    b,
    h,
    n,
    d,
    iters=30,
    warmup=10,
    dtype=torch.float16,
):
    flops = attention_flops(b, h, n, d)

    Q = torch.randn(b, h, n, d, device="cuda", dtype=dtype)
    K = torch.randn(b, h, n, d, device="cuda", dtype=dtype)
    V = torch.randn(b, h, n, d, device="cuda", dtype=dtype)

    # warmup
    for _ in range(warmup):
        fn(Q, K, V)
    torch.cuda.synchronize()

    # timing
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(Q, K, V)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    t = (t1 - t0) / iters

    return {
        "name": name,
        "time": t,
        "tflops": flops / t / 1e12,
    }


def sweep_seq_len(
    sizes,
    ops,
    b=4,
    h=8,
    d=64,
    iters=30,
    warmup=10,
):
    """
    sizes: sequence lengths
    ops: {name: function}
    """

    results = {name: [] for name in ops}

    for n in sizes:
        print(f"\nseq_len = {n}")
        for name, fn in ops.items():
            res = run_attention_benchmark(
                name=name,
                fn=fn,
                b=b,
                h=h,
                n=n,
                d=d,
                iters=iters,
                warmup=warmup,
            )
            print(f"{name}: {res['tflops']:.2f} TFLOPS")
            results[name].append(res["tflops"])

    return results

def plot_attention_benchmark(
    sizes,
    perf_data,
    out_dir="results",
    out_name="attention_benchmark.png",
    title="Attention Performance",
):
    """
    sizes: [seq_len]
    perf_data: {name: [tflops]}
    """

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)

    plt.figure(figsize=(8, 6))

    for name, tflops in perf_data.items():
        plt.plot(sizes, tflops, marker="o", label=name)

    plt.xlabel("Sequence Length")
    plt.ylabel("TFLOPS")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"\nSaved figure to: {out_path}")
