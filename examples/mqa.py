from attention.mqa import mqa
import torch


def main():
    b, h, n, d = 16, 8, 32, 64
    dtype = torch.float32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    Q = torch.randn(b, h, n, d, device=device, dtype=dtype)
    K = torch.randn(b, n, d, device=device, dtype=dtype)
    V = torch.randn(b, n, d, device=device, dtype=dtype)

    output, attention_weights = mqa(Q, K, V, causal=True, return_attn=True)

    print(f"output shape: {output.shape}")
    print(f"attention_weights shape: {attention_weights.shape}")


if __name__ == "__main__":
    main()
