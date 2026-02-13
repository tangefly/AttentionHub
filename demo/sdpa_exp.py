from attention import scaled_dot_product_attention
import torch

b, h, n, d = 16, 8, 32, 64
dtype = torch.float32

Q = torch.randn(b, h, n, d, device="cuda", dtype=dtype)
K = torch.randn(b, h, n, d, device="cuda", dtype=dtype)
V = torch.randn(b, h, n, d, device="cuda", dtype=dtype)

output, attention_weights = scaled_dot_product_attention(Q, K, V)

print(f"output shape: {output.shape}")
print(f"attention_weights shape: {attention_weights.shape}")
