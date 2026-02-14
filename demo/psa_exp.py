from attention import PredictSparseAttention
import torch

b, n, d, k, topk = 16, 256, 1024, 32, 32
dtype = torch.float32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

X = torch.randn(b, n, d, device=device, dtype=dtype)

psa = PredictSparseAttention(d, k, topk).to(device)

output = psa(X)

print(f"output: {output.shape}")