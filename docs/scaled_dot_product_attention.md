## 1 计算公式

$$
Attention(Q,K,V) = softmax(\frac{QK^{T}}{\sqrt{d_{k}} } )V
$$

![image1](https://github.com/user-attachments/assets/869133e8-30ee-4056-aea3-0eee689c638c)

在计算 $QK^{T}$ 时，每个 $q$ 都会和每个 $k$ 计算点积，而 $a \cdot b = \left | a \right | \left | b \right | cos(\theta )$，即每个 $q$ 和 $k$ 的乘积等于两个向量的余弦相似度和两个向量的乘积。所以这种矩阵乘法可以得到每个 $q$ 和 $k$ 的相似度分数，再经过 $softmax$ 可以得到每个 $q$ 对 $k$ 的重要性权重。

在维度非常大时，可能会导致点积的值非常大，根据 $softmax$ 的计算特性，有可能会导致重要性权重分布极端，所以添加了缩放因子去消除这一局限。

## 2 优点

- 可以计算每个 $q$ 和 $k$ 的注意力，效果好

## 3 缺点

- 计算复杂度高，对于 $token$ 数量 $n$ 来说，$n$ 个 $q$ 要和 $n$ 个 $k$ 计算点积，随着 $n$ 的增长，复杂度呈现 $O(n^2)$。
- 由于计算复杂度呈现 $O(n^2)$，计算复杂度会导致 LLM 对长文本的支持减弱。

## 4 Demo

```
from attention import scaled_dot_product_attention
import torch

b, h, n, d = 16, 8, 32, 64
dtype = torch.float32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

Q = torch.randn(b, h, n, d, device=device, dtype=dtype)
K = torch.randn(b, h, n, d, device=device, dtype=dtype)
V = torch.randn(b, h, n, d, device=device, dtype=dtype)

output, attention_weights = scaled_dot_product_attention(Q, K, V)

print(f"output shape: {output.shape}")
print(f"attention_weights shape: {attention_weights.shape}")
```