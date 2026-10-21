from attention.gqa import gqa
import torch


def main():
    b, qh, kvh, n, d = 16, 8, 4, 32, 64
    dtype = torch.float32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    Q = torch.randn(b, qh, n, d, device=device, dtype=dtype)
    K = torch.randn(b, kvh, n, d, device=device, dtype=dtype)
    V = torch.randn(b, kvh, n, d, device=device, dtype=dtype)

    output, attention_weights = gqa(Q, K, V, causal=True, return_attn=True)

    print(f"output shape: {output.shape}")
    print(f"attention_weights shape: {attention_weights.shape}")


if __name__ == "__main__":
    main()
