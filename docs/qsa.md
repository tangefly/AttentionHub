# Qwen Sparse Attention (QSA)

## 1 基本理论

QSA 是 Qwen4-Exp 中使用的稀疏注意力层。它不直接对全部 token 计算注意力，而是先用一个轻量的 indexer 选出每个 query 需要关注的 token 子集，再在子集上做标准注意力。

indexer 的流程：

1. 把当前 query 可见的 key 按 `compress_ratio` 个 token 分块。
2. 对每个块内的 key 做均值池化，得到块级 key 表示。
3. 用 query 与块级 key 打分，取分数最高的 `block_topk` 个块。
4. `block_topk = indexer_budget // compress_ratio`，选中的块展开回 token 级别。
5. 未组成完整块的尾部 token 始终保留。
6. 把选中的 token 掩码与因果掩码合并，再做普通注意力。

因此每个 query 保留的 token 数不超过：

```
indexer_budget + indexer_compress_ratio - 1
```

## 2 复杂度

设序列长度为 `N`，主注意力 head dim 为 `D`，indexer head dim 为 `d_i`，块数为 `N / r`：

- indexer 打分：`O(N * (N / r) * d_i)`
- 主注意力：`O(N * budget * D)`
- 相比全注意力 `O(N^2 * D)`，当 `budget << N` 时显著降低主注意力开销。

## 3 优点与缺点

优点：

- block 级选择比 token 级 top-k 更容易硬件加速。
- indexer 维度小，额外开销可控。
- 预算 `indexer_budget` 可以直接控制稀疏度。

缺点：

- indexer 的块级池化会损失细粒度信息。
- 需要额外的 indexer 参数和打分计算。
- 实现上包含逐 query 的选择循环，参考实现不适合直接用于生产推理。

## 4 接口

QSA 是一个带参数的层，不是纯函数算子：

```python
from attention.qsa import QwenSparseAttention
from attention.rope import build_rope_cache_batched

layer = QwenSparseAttention(
    head_dim=64,
    hidden_size=256,
    num_attention_heads=4,
    num_key_value_heads=2,
    attention_dropout=0.0,
    attention_bias=False,
    rms_norm_eps=1e-6,
    indexer_n_heads=2,
    indexer_kv_heads=1,
    indexer_head_dim=32,
    indexer_budget=32,
    indexer_compress_ratio=8,
)

cos, sin = build_rope_cache_batched(
    batch, seq_len, rotary_dim, device=hidden_states.device, dtype=hidden_states.dtype
)
output, attention_weights = layer(hidden_states, (cos, sin), attention_mask)
```

约束：

- `indexer_kv_heads` 必须为 1。
- `indexer_budget` 必须能被 `indexer_compress_ratio` 整除。
- `attention_mask` 为 4D，bool（`True` = 可见）或 additive float。
- `position_embeddings` 是 `(cos, sin)`，形状 `(B, N, rotary_dim)`，可用 [attention/rope.py](../attention/rope.py) 构造。

完整可运行示例见 [examples/qsa.py](../examples/qsa.py)。

## 5 实现

- 参考实现：[attention/qsa/layer.py](../attention/qsa/layer.py)
- 示例：[examples/qsa.py](../examples/qsa.py)
- 测试：[tests/test_qsa.py](../tests/test_qsa.py)
