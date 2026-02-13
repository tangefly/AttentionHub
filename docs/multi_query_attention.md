## 1 计算公式

$$
Attention(Q,K,V) = softmax(\frac{QK^{T}}{\sqrt{d_{k}} } )V
$$

MQA 的计算公式和 SDPA 的计算公式一模一样，区别在于，MQA 的多个 head 共享相同的 $K$ 和 $V$ 。这样可以减少 $K,V$ 的显存占用，但是计算量未必减少，因为每个注意力头仍然都要计算注意力分数，只不过这里是对同一份 $K,V$ 进行计算。 

## 2 优点

- 可以计算每个 $q$ 和 $k$ 的注意力，效果好
- 减少了 $K,V$ 的显存占用

## 3 缺点

- 计算复杂度高，对于 $token$ 数量 $n$ 来说，$n$ 个 $q$ 要和 $n$ 个 $k$ 计算点积，随着 $n$ 的增长，复杂度呈现 $O(n^2)$。
- 由于计算复杂度呈现 $O(n^2)$，计算复杂度会导致 LLM 对长文本的支持减弱。
- 由于多个注意力头共享同一份 $K,V$，可能导致模型表达能力下降，模型训练更慢收敛。

## 4 Demo

```
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
```