from attention import factorized_attention
import torch

b, h, n, d, l = 16, 8, 32, 64, 2
dtype = torch.float32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

Q = torch.randn(b, h, n, d, device=device, dtype=dtype)
K = torch.randn(b, h, n, d, device=device, dtype=dtype)
V = torch.randn(b, h, n, d, device=device, dtype=dtype)

output = factorized_attention(Q, K, V, l)

print(f"output shape: {output.shape}")
