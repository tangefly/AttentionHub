# Multi-Head Latent Attention (MLA)

## 1 基本理论

MLA 是 DeepSeek-V2 使用的注意力层。它不保存每个 head 的完整 K/V，而是把 K/V
压缩成一个所有 head 共享的低秩 latent，在计算注意力前再上投影回每个 head。

对一个 token：

1. 用 `kv_a_proj_with_mqa` 把 hidden state 投影成
   `[kv_lora_rank, qk_rope_head_dim]`：
   - 前 `kv_lora_rank` 维是 KV latent，先做 RMSNorm；
   - 后 `qk_rope_head_dim` 维是解耦的 RoPE key（`k_pe`）。
2. `kv_b_proj` 把归一化后的 latent 展开成每个 head 的
   `[qk_nope_head_dim, v_head_dim]`，切分成 `k_nope` 和 `value`。
3. key 由 `k_nope` 与所有 head 共享的 `k_pe` 拼接而成：
   `key = [k_nope, k_pe]`。
4. query 同样拆成 `q_nope` 和 `q_pe`；RoPE 只作用在 `q_pe` / `k_pe` 上。
5. 在 `[k_nope, k_pe]` 与 `[q_nope, q_pe]` 上做普通 scaled dot-product
   attention。

如果启用 query 低秩压缩（`q_lora_rank` 不为 `None`），query 先经过
`q_a_proj -> RMSNorm -> q_b_proj`；否则直接用 `q_proj`。

RoPE 的通道布局由 `rope_layout` 控制：

- `"interleaved"`（默认）：DeepSeek-V2 原始权重使用的复数 RoPE，相邻的
  `(2i, 2i + 1)` 通道组成一个旋转对。
- `"half"`：仓库统一的 `rotate_half` 布局（与 Llama 相同），前一半通道与
  后一半通道配对。

两种模式都复用 [attention/rope.py](../attention/rope.py) 的
`build_rope_cache_batched` 和 `apply_rotary_pos_emb`。`"interleaved"` 模式在
调用共享实现前把相邻通道重排成 half 布局，调用后再还原，因此不需要额外的
RoPE kernel，同时保持与 DeepSeek-V2 权重一致的旋转语义。

## 2 复杂度与 KV cache

设序列长度为 `N`，head 数为 `H`，每个 head 的 K/V 维度为 `D`。

| 方案 | 每 token KV cache | 注意力计算 |
| --- | --- | --- |
| MHA | `2 * H * D` | `O(N^2 * H * D)` |
| MLA | `kv_lora_rank + qk_rope_head_dim` | `O(N^2 * H * (qk_head_dim + v_head_dim))` |

由于 `kv_lora_rank + qk_rope_head_dim` 通常远小于 `2 * H * D`，MLA 的主要收益
是显著减小推理时的 KV cache。query 压缩（`q_lora_rank`）则主要降低训练/推理的
激活显存。

参考实现为了可读性直接展开 latent 再做注意力，没有实现增量 KV cache；上表描述
的是该结构在推理引擎中的理论收益。

## 3 优点与缺点

优点：

- KV cache 与 head 数解耦，长上下文推理的显存占用低。
- 解耦 RoPE 只旋转一小部分维度，位置信息与内容信息分离。
- 与标准 attention 的数学形式兼容，容易和现有 kernel 对接。

缺点：

- 需要额外的 down/up projection，训练时计算量高于普通 GQA。
- latent 是信息瓶颈，rank 太小会损失表达能力。
- 解耦 RoPE 的通道顺序与部分预训练权重不兼容，需要按实现约定对齐。

## 4 接口

MLA 是一个带参数的层，不是纯函数算子：

```python
import torch

from attention.mla import DeepseekV2Attention
from attention.rope import build_rope_cache_batched

batch, seq_len, hidden_size = 2, 64, 256
layer = DeepseekV2Attention(
    hidden_size=hidden_size,
    num_attention_heads=8,
    kv_lora_rank=32,
    qk_nope_head_dim=32,
    qk_rope_head_dim=16,
    v_head_dim=32,
    q_lora_rank=64,
    attention_dropout=0.0,
    attention_bias=False,
    rms_norm_eps=1e-6,
    rope_layout="interleaved",
).eval()

hidden_states = torch.randn(batch, seq_len, hidden_size)
cos, sin = build_rope_cache_batched(
    batch, seq_len, 16, device=hidden_states.device, dtype=hidden_states.dtype
)

# attention_mask 可省略；省略时默认使用 causal mask。
output, attention_weights = layer(hidden_states, (cos, sin))
```

约束与约定：

- `qk_rope_head_dim` 必须为偶数。
- `q_lora_rank=None` 时使用直接的 `q_proj`，否则使用 query 低秩压缩。
- `rope_layout` 可选 `"interleaved"`（默认，匹配 DeepSeek-V2 权重）或
  `"half"`（匹配仓库/Llama 的 `rotate_half` 约定）。
- `position_embeddings` 是 `(cos, sin)`，形状为
  `(B, seq_len, qk_rope_head_dim)`；也接受 `(1, seq_len, d)` 或
  `(seq_len, d)`。超出当前输入长度的部分会被截取到最后 `seq_len` 个位置。
- `attention_mask` 为 bool（`True` = 可见）或 additive float，需可广播到
  `(B, H, seq_len, seq_len)`。传入显式 mask 时不会再额外叠加 causal mask。
- 返回 `(output, attention_weights)`，形状分别为
  `(B, seq_len, hidden_size)` 和 `(B, H, seq_len, seq_len)`。
- KV latent 对所有 head 共享，因此本层没有 `num_key_value_heads` 参数。

完整可运行示例见 [examples/mla.py](../examples/mla.py)。

## 5 实现

- 参考实现：[attention/mla/layer.py](../attention/mla/layer.py)
- 共享 RMSNorm：[attention/norm.py](../attention/norm.py)
- 共享 RoPE：[attention/rope.py](../attention/rope.py)
- RoPE 布局：[docs/rope.md](rope.md)
- 示例：[examples/mla.py](../examples/mla.py)
- 测试：[tests/test_mla.py](../tests/test_mla.py)
