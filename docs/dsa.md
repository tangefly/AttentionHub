# DeepSeek Sparse Attention (DSA)

## 1 基本理论

DSA（DeepSeek Sparse Attention）是 DeepSeek-V3.2-Exp 使用的注意力层。它保留
MLA 的低秩 KV 压缩和解耦 RoPE，在 MLA 之上增加一个轻量的 **indexer**：每个
query 只保留分数最高的 `index_topk` 个 key，主注意力在选中的子集上计算。

### 1.1 主注意力（MLA）

主注意力与 [MLA](mla.md) 完全一致：

```
q_t      = [q_nope_t, q_pe_t]          # 低秩 query 投影 + 解耦 RoPE
c_t      = W_DKV x_t                    # [kv_lora_rank + qk_rope_head_dim]
k_t, v_t = W_UK c_t, W_UV c_t
```

不同之处只在 softmax 之前：DSA 把未入选的 key 屏蔽掉。

### 1.2 Indexer

indexer 用少量 head 和较小的 head dim 估计每个 key 的重要性。对 token `t`：

```
q_i,t   = W_q,i · q_resid_t                     # 每个 index head 一个 query
k_t     = LayerNorm(W_k x_t)                    # 所有 index head 共享一个 key
q_i,t   = RoPE(q_i,t),  k_t = RoPE(k_t)         # 只旋转前 index_rope_head_dim 维
s_i,t   = relu(q_i,t · k_t) * index_head_dim^-0.5
w_t,i   = (W_w x_t)_i * index_n_heads^-0.5
score_t = Σ_i w_t,i · s_i,t
I_t     = topk(score_t, index_topk)
```

要点：

- `q_resid` 是 MLA 的 query latent，即 `q_a_layernorm(q_a_proj(x))`；没有 query
  压缩时直接使用 hidden states。
- indexer 的 key 用 **LayerNorm**（不是 RMSNorm），这与开源权重一致。
- indexer 的 RoPE 使用仓库统一的 **half-split** 布局；主 MLA 的 RoPE 布局由
  `rope_layout` 控制（默认 `interleaved`，匹配 DeepSeek 权重），两者互不影响。
- 打分先 `relu` 再加权求和，因此只有正相关的 index head 会贡献分数。
- 最后按因果/ padding mask 把不可见位置置为 `-inf`，再做 top-k。

DeepSeek-V3.2-Exp 的典型配置为：`index_n_heads=64`、
`index_head_dim=128`、`index_rope_head_dim=64`、`index_topk=2048`。

### 1.3 稀疏 mask

indexer 返回 `(B, S, topk)` 的 key 下标。层内把它转成布尔 mask `keep`：

```
keep[b, s, t] = t ∈ I_{b,s}
```

再与用户 mask（或默认 causal mask）合并：

- bool mask：`mask & keep`；
- float mask：未入选位置填 `finfo.min`。

因此最终每个 query 最多 attend `index_topk` 个 key；可见 key 少于 `index_topk`
时实际保留的 key 更少。top-k 在 mask 之后执行，所以 indexer 不会选中未来 token
或 padding token 参与最终 softmax。

## 2 复杂度

设序列长度为 `N`，主注意力 head 数为 `H`、head dim 为 `D`，indexer head 数为
`H_i`、head dim 为 `d_i`：

| 部分 | 计算量 | 说明 |
| --- | --- | --- |
| indexer | `O(N^2 · H_i · d_i)` | 仍是二次复杂度，但 `H_i·d_i` 远小于主注意力 |
| 主注意力 | `O(N · topk · H · D)` | 由 `O(N^2 · H · D)` 降为线性（固定 topk） |
| KV cache | `kv_lora_rank + qk_rope_head_dim` | 与 MLA 相同，不含 indexer cache |

indexer 自身仍是 `O(N^2)`，这是 DSA 的主要剩余开销；实际系统中会把 indexer 的
key 量化到 FP8 并使用融合 kernel，以降低这部分成本。参考实现为了可读性逐 token
构建 mask，不追求 kernel 级性能。

## 3 优点与缺点

优点：

- 主注意力从二次降为 `O(N · topk)`，长上下文收益明显。
- 复用 MLA 的 KV cache 压缩，显存占用不变。
- indexer 很轻（head 数少、维度小），额外参数和计算可控。
- 与 MLA/FlashMLA 的接口兼容，索引可以直接传给稀疏 MLA kernel。

缺点：

- indexer 打分仍是 `O(N^2)`，序列极长时它会成为瓶颈。
- top-k 选择不可导；训练 indexer 需要额外的蒸馏/对齐损失，本仓库的参考层只
  实现前向选择。
- 稀疏选择会损失一部分长距离信息，`index_topk` 是精度与效率的权衡。
- 参考实现返回稠密 attention weights，长序列下显存占用仍按 `N^2` 计。

## 4 接口

DSA 是一个带参数的层，采用与 MLA 相同的投影和 RoPE 约定，但实现独立，
不是 `MultiHeadLatentAttention` 的派生类：

```python
import torch

from attention.dsa import DeepseekSparseAttention
from attention.rope import build_rope_cache_batched

batch, seq_len, hidden_size = 2, 512, 256

layer = DeepseekSparseAttention(
    hidden_size=hidden_size,
    num_attention_heads=8,
    kv_lora_rank=64,
    qk_nope_head_dim=32,
    qk_rope_head_dim=16,
    v_head_dim=32,
    q_lora_rank=128,
    index_n_heads=4,
    index_head_dim=32,
    index_rope_head_dim=16,
    index_topk=32,
    attention_dropout=0.0,
    attention_bias=False,
    rms_norm_eps=1e-6,
    rope_layout="interleaved",
).eval()

hidden_states = torch.randn(batch, seq_len, hidden_size)
cos, sin = build_rope_cache_batched(
    batch, seq_len, 16, device=hidden_states.device, dtype=hidden_states.dtype
)

# mask 可省略，省略时默认 causal。
output, attention_weights = layer(hidden_states, (cos, sin))
```

约束与约定：

- `index_topk` 会被截断到 KV 长度，因此 `index_topk >= seq_len` 时退化为稠密
  MLA。
- `index_rope_head_dim` 默认为 `qk_rope_head_dim`，必须为偶数且不超过
  `index_head_dim`。
- `q_lora_rank=None` 时主注意力使用直接 `q_proj`，indexer 直接以 hidden states
  作为 query latent，此时 `query_dim == hidden_size`。
- `position_embeddings` 是 `(cos, sin)`，形状为
  `(B, seq_len, qk_rope_head_dim)`；indexer 复用同一组表，并把
  `index_rope_head_dim` 作为旋转维。超出当前输入长度的部分会被截取到最后
  `seq_len` 个位置。
- `attention_mask` 为 bool（`True` = 可见）或 additive float，需可广播到
  `(B, 1, seq_len, seq_len)`；也接受 2D 的 `(seq_len, seq_len)`。省略时默认
  causal。
- 返回 `(output, attention_weights)`，形状分别为
  `(B, seq_len, hidden_size)` 和 `(B, H, seq_len, seq_len)`；未被选中的位置
  权重为 0。
- 本层不实现增量 KV cache（`past_key_values`），与仓库中其他参考层一致。
- 需要单独检查索引时，可以调用 `layer.indexer`：

```python
q_resid = layer.q_a_layernorm(layer.q_a_proj(hidden_states))
indices = layer.indexer(hidden_states, q_resid, (cos, sin))
# indices: (B, seq_len, min(index_topk, seq_len)), int64
```

完整可运行示例见 [examples/dsa.py](../examples/dsa.py)。

## 5 实现

- 参考实现：[attention/dsa/layer.py](../attention/dsa/layer.py)
  - `DeepseekSparseIndexer`：indexer 打分与 top-k；
  - `DeepseekSparseAttention`：MLA 主注意力 + 稀疏 mask。
- MLA 参考实现（参数命名与 RoPE 约定保持一致）：
  [attention/mla/layer.py](../attention/mla/layer.py)
- 共享 RoPE：[attention/rope.py](../attention/rope.py)
- 共享 RMSNorm：[attention/norm.py](../attention/norm.py)
- 示例：[examples/dsa.py](../examples/dsa.py)
- 测试：[tests/test_dsa.py](../tests/test_dsa.py)
- 参考：DeepSeek-V3.2-Exp,
  <https://github.com/deepseek-ai/DeepSeek-V3.2-Exp>
