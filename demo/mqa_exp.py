from attention import multi_query_attention
import torch

b, h, n, d = 16, 8, 32, 64
dtype = torch.float32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

Q = torch.randn(b, h, n, d, device=device, dtype=dtype)
K = torch.randn(b, n, d, device=device, dtype=dtype)
V = torch.randn(b, n, d, device=device, dtype=dtype)

output, attention_weights = multi_query_attention(Q, K, V)

print(f"output shape: {output.shape}")
print(f"attention_weights shape: {attention_weights.shape}")
