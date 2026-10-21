from attention.fta import fta
import torch


def main():
    b, h, n, d = 16, 8, 32, 64
    l = 2
    dtype = torch.float32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    Q = torch.randn(b, h, n, d, device=device, dtype=dtype)
    K = torch.randn(b, h, n, d, device=device, dtype=dtype)
    V = torch.randn(b, h, n, d, device=device, dtype=dtype)

    output = fta(Q, K, V, l)

    print(f"output shape: {output.shape}")


if __name__ == "__main__":
    main()
