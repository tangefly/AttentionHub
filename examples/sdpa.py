from attention.sdpa import sdpa
import torch


def main():
    b, h, n, d = 16, 8, 32, 64
    dtype = torch.float32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    Q = torch.randn(b, h, n, d, device=device, dtype=dtype)
    K = torch.randn(b, h, n, d, device=device, dtype=dtype)
    V = torch.randn(b, h, n, d, device=device, dtype=dtype)

    output = sdpa(Q, K, V, causal=True)
    reference, attention_weights = sdpa(
        Q, K, V, causal=True, backend="reference", return_attn=True
    )

    print(f"output shape: {output.shape}")
    print(f"attention_weights shape: {attention_weights.shape}")
    print(f"max |auto - reference|: {(output - reference).abs().max().item():.2e}")


if __name__ == "__main__":
    main()
