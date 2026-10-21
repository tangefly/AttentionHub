# RoPE 工具

RoPE（Rotary Position Embedding）的 cos/sin 构造和旋转应用集中在
[attention/rope.py](../attention/rope.py)，QSA 和示例都复用这里的实现。

## 1 构造 cos/sin

```python
from attention.rope import build_rope_cache, build_rope_cache_batched

# (seq_len, rotary_dim)
cos, sin = build_rope_cache(seq_len, rotary_dim, device=device)

# (batch, seq_len, rotary_dim)，适合 (B, H, N, D) 的 q/k
cos, sin = build_rope_cache_batched(
    batch_size, seq_len, rotary_dim, device=device, dtype=q.dtype
)
```

参数：

| 参数 | 说明 |
| --- | --- |
| `seq_len` | 位置数量 |
| `dim` / `rotary_dim` | 旋转维度，必须为偶数 |
| `base` | RoPE theta base，默认 `10000.0` |
| `device` | 返回 tensor 的设备 |
| `dtype` | 返回 tensor 的 dtype；频率内部始终用 float32 计算 |

## 2 应用旋转

```python
from attention.rope import apply_rotary_pos_emb

q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
```

`q` / `k` 通常是 `(batch, heads, seq_len, head_dim)`；`cos` / `sin` 为
`(batch, seq_len, rotary_dim)` 或 `(1, seq_len, rotary_dim)`。
`cos` / `sin` 会自动转换到 `q` 的 device/dtype。

当 `rotary_dim < head_dim` 时，超出 `rotary_dim` 的部分保持不变（partial RoPE）。

## 3 QSA 中的用法

```python
from attention.qsa import QwenSparseAttention
from attention.rope import build_rope_cache_batched

cos, sin = build_rope_cache_batched(
    batch, seq_len, rotary_dim, device=device, dtype=hidden_states.dtype
)
output, attention_weights = qsa(hidden_states, (cos, sin), attention_mask)
```

## 4 MLA 中的用法

MLA 的 RoPE 只作用在解耦的 query/key 分量上，因此 `rotary_dim` 等于
`qk_rope_head_dim`。MLA 默认使用 DeepSeek 的 interleaved 通道布局，内部会先把
相邻通道重排成这里的 half 布局，再调用共享的 `apply_rotary_pos_emb`；
`rope_layout="half"` 则直接使用 half 布局：

```python
from attention.mla import DeepseekV2Attention
from attention.rope import build_rope_cache_batched

cos, sin = build_rope_cache_batched(
    batch, seq_len, qk_rope_head_dim, device=device, dtype=hidden_states.dtype
)
output, attention_weights = mla(hidden_states, (cos, sin), attention_mask)
```

完整示例见 [examples/mla.py](../examples/mla.py)。

## 5 测试

- [tests/test_rope.py](../tests/test_rope.py)：形状、`cos² + sin² = 1`、norm 保持、partial RoPE、dtype、异常输入。
- [tests/test_qsa.py](../tests/test_qsa.py)：QSA 使用共享 RoPE 后与 dense attention 对齐。
- [tests/test_mla.py](../tests/test_mla.py)：MLA 使用共享 RoPE 后与手工参考实现对齐。
