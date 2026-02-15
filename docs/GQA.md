## 1 计算公式

$$
Attention(Q,K,V) = softmax(\frac{QK^{T}}{\sqrt{d_{k}} } )V
$$

GQA 的计算公式和 SDPA(MHA) 的计算公式还是一模一样，区别在于 Q,K,V 这三个张量的 shape 不是对齐的，GQA 的多个 head 共享相同的 $K$ 和 $V$，并且 $K,V$ 的数量不是唯一，而是多余 1 个。所以 GQA 是 MHA 和 MQA 的一个折中方案。

## 2 优点

- 可以计算每个 $q$ 和 $k$ 的注意力，效果好
- 减少了 $K,V$ 的显存占用
- 即保留了多个 $K,V$ 又减少了计算复杂度，在效果和速度上实现了折中

## 3 缺点

- 计算复杂度高，对于 $token$ 数量 $n$ 来说，$n$ 个 $q$ 要和 $n$ 个 $k$ 计算点积，随着 $n$ 的增长，复杂度呈现 $O(n^2)$。
- 由于计算复杂度呈现 $O(n^2)$，计算复杂度会导致 LLM 对长文本的支持减弱。

## 4 Demo

```python
from attention import grouped_query_attention
import torch

b, qh, kvh, n, d = 16, 8, 4, 32, 64
dtype = torch.float32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

Q = torch.randn(b, qh, n, d, device=device, dtype=dtype)
K = torch.randn(b, kvh, n, d, device=device, dtype=dtype)
V = torch.randn(b, kvh, n, d, device=device, dtype=dtype)

output, attention_weights = grouped_query_attention(Q, K, V)

print(f"output shape: {output.shape}")
print(f"attention_weights shape: {attention_weights.shape}")
```