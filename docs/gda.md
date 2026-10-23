# Gated Delta Attention (GDA)

## 1 基本理论

GDA（Gated Delta Attention，也写作 Gated DeltaNet / GDN）是一种线性注意力：
它不显式构造 `N × N` 的注意力矩阵，而是为每个 head 维护一个
`d_k × d_v` 的矩阵状态 `S`，并按 token 递推。Qwen3.5 的
`linear_attention` 层就是 GDA 块。

### 1.1 门控 delta 递推

对第 `t` 个 token，记 query、key、value 分别为 `q_t, k_t, v_t`，log 衰减为
`g_t ≤ 0`，写强度为 `β_t ∈ (0, 1)`：

```
S_t = S_{t-1} · exp(g_t)                     # 门控遗忘
δ_t = (v_t - S_tᵀ k_t) · β_t                 # delta 修正
S_t = S_t + k_t · δ_tᵀ                       # rank-1 写入
o_t = S_tᵀ q_t                               # 读取输出
```

与普通线性注意力 `S_t = S_{t-1} + k_t v_tᵀ` 相比，delta rule 只写入当前状态
**还预测不出来的部分** `v_t - S_tᵀ k_t`，因此不会把新旧信息简单叠加；`β_t`
控制写入强度，`exp(g_t)` 控制遗忘速度。当 `g_t = 0, β_t = 1` 时退化为经典
delta rule，`β_t = 0` 时状态冻结。

### 1.2 Qwen3.5 的 GDA 块

完整的层在递推前还有一次局部卷积，在递推后还有输出门：

```
qkv = W_qkv x                          # key | key | value
qkv = causal_conv1d(qkv)               # depthwise, kernel 4
q, k, v = split(qkv)
z = W_z x                              # 输出门
β = sigmoid(W_b x)                     # 写强度
g = -exp(A_log) · softplus(W_a x + dt_bias)   # log 衰减
o = gated_delta_rule(q, k, v, g, β)
y = W_o (RMSNorm(o) · silu(z))
```

- 因果卷积让每个位置先融合最近 `conv_kernel_size - 1` 个 token，弥补线性
  注意力没有显式 token 交互的问题。
- query/key 在 kernel 内做 L2 归一化，query 再乘以 `d_k^-0.5`。
- 输出先做 RMSNorm，再乘 SiLU 门 `silu(z)`，最后过 `out_proj`。
- `num_v_heads` 可以大于 `num_k_heads`（GQA 式分组），此时 query/key head 会
  按组重复。

### 1.3 分块（chunked）并行

逐 token 递推无法利用矩阵乘法的并行性。GDA 使用 UT（upper-triangular）
变换把序列切成 `C` 个 token 的块，对块内做矩阵运算，只对块做短序列扫描。
设块内累计衰减 `G_i = Σ_{j≤i} g_j`，成对衰减
`D_ij = exp(G_i - G_j)`（`i ≥ j`），并记
`K_β = K · diag(β)`、`V_β = V · diag(β)`、`L = tril(K_β Kᵀ ⊙ D, -1)`：

```
U       = (I + L)⁻¹ V_β
K_cum   = (I + L)⁻¹ (K_β ⊙ exp(G))
V_new   = U - K_cum S_0
O       = (Q ⊙ exp(G)) S_0 + [(Q Kᵀ ⊙ D) V_new]
S_C     = exp(G_C) S_0 + (K ⊙ exp(G_C - G))ᵀ V_new
```

`(I + L)` 是单位下三角矩阵，用前向代入（`torch.linalg.solve_triangular`）
求解。这样块内复杂度从 `O(C²)` 的串行递推变成若干次 matmul，所有块共享同一
套形状；块之间只保留 `S` 的串行扫描。解码时 `T = 1`，直接使用逐 token 递推
更划算，层内会自动切换。

## 2 复杂度

设序列长度为 `N`，head 数为 `H`，key/value head dim 为 `d_k, d_v`，块大小为
`C`：

| 部分 | 计算量 | 说明 |
| --- | --- | --- |
| 块内 | `O(N · C · (d_k + d_v))` | `C²` 的 matmul / 三角求解 |
| 块间状态读写 | `O(N · d_k · d_v)` | 每个 token 读写一次矩阵状态 |
| 卷积 | `O(N · H · (d_k + d_v) · conv_kernel)` | depthwise，可忽略 |
| 状态显存 | `O(H · d_k · d_v)` | 与序列长度无关 |
| 对比 softmax 注意力 | `O(N² · d)` | 二次复杂度 |

因此 GDA 的推理成本随序列长度线性增长，KV cache 也退化为固定大小的
recurrent state。训练时块内可以并行，但块间仍是短扫描；实际框架会用融合
kernel（FLA）进一步降低开销。

注意：本仓库的纯 PyTorch 分块实现会显式物化每个块的 `C × C` 衰减矩阵，
额外显存约为 `O(B · H · N · C)`；超长序列建议使用融合 kernel。

## 3 优点与缺点

优点：

- 长序列下计算和显存都是线性的，适合超长上下文。
- delta rule 让状态写入更有选择性，比纯累加式线性注意力有更好的记忆能力。
- 门控衰减允许模型按 token 自适应地遗忘。
- 卷积 + 输出门增强了局部建模和表达能力。
- 推理时只需要一个固定大小的矩阵状态，不需要随长度增长的 KV cache。

缺点：

- 训练并行度不如 softmax 注意力：块间仍有一个串行扫描。
- 递推对数值精度和 kernel 实现较敏感，长序列训练通常需要 fp32 状态。
- 与标准注意力的权重可视化不兼容，本仓库的实现不返回 attention weights。
- 卷积和门控引入额外参数；权重与具体 checkpoint（Qwen3.5）绑定。

## 4 接口

### 4.1 纯函数

`reference` 是逐 token 的可读 oracle，`torch` 是分块实现；两者都使用
`(batch, seq_len, num_heads, dim)` 布局：

```python
import torch

from attention.gda import gated_delta_rule

batch, seq_len, heads, d_k, d_v = 2, 128, 4, 32, 32
q = torch.randn(batch, seq_len, heads, d_k)
k = torch.randn(batch, seq_len, heads, d_k)
v = torch.randn(batch, seq_len, heads, d_v)
g = -torch.rand(batch, seq_len, heads)          # log decay <= 0
beta = torch.rand(batch, seq_len, heads)        # write strength

out, final_state = gated_delta_rule(
    q, k, v, g, beta,
    output_final_state=True,
    use_qk_l2norm_in_kernel=True,
    backend="torch",                            # 或 "reference"
)
# out: (batch, seq_len, heads, d_v)
# final_state: (batch, heads, d_k, d_v), float32
```

`chunk_size` 只影响 `torch` 后端；`initial_state` 可以传入上一段序列的状态，
实现分段续算。`backend="auto"` 按仓库约定选择 `torch`。

`torch` 后端是 `transformers` 中 `torch_chunk_gated_delta_rule` /
`torch_recurrent_gated_delta_rule` 的逐行移植：默认参数下与 `transformers`
的 eager fallback 逐位一致（见 [tests/test_gda.py](../tests/test_gda.py)），
并额外接受 `transformers` / FLA 调用点传入的 kernel 关键字（如
`cu_seqlens`，eager 路径会忽略）。`scale` 是本仓库的扩展，其默认值
`head_k_dim ** -0.5` 与 `transformers` 相同。

### 4.2 GDA 层

从 Qwen3.5 的 config 构造时，字段一一对应：

```python
layer = GatedDeltaNet(
    hidden_size=config.hidden_size,
    num_v_heads=config.linear_num_value_heads,
    num_k_heads=config.linear_num_key_heads,
    head_k_dim=config.linear_key_head_dim,
    head_v_dim=config.linear_value_head_dim,
    conv_kernel_size=config.linear_conv_kernel_dim,
    rms_norm_eps=config.rms_norm_eps,
    activation=config.hidden_act,
)
```

`GatedDeltaNet` 是去掉 `transformers` 依赖后的 Qwen3.5 GDA 块，参数名与
HuggingFace checkpoint 保持一致：

```python
import torch

from attention.gda import GatedDeltaNet

layer = GatedDeltaNet(
    hidden_size=512,
    num_v_heads=8,
    num_k_heads=4,
    head_k_dim=64,
    head_v_dim=64,
    conv_kernel_size=4,
    rms_norm_eps=1e-6,
    activation="silu",
).eval()

x = torch.randn(2, 64, 512)
out = layer(x)                                  # (2, 64, 512)
```

增量解码：

```python
with torch.no_grad():
    out, state = layer(x, use_cache=True)

    step_out, state = layer(x[:, -1:], past_state=state, use_cache=True)
    # step_out: (2, 1, 512)
```

`state` 是 :class:`attention.gda.GatedDeltaNetState`：

| 字段 | 形状 | 说明 |
| --- | --- | --- |
| `conv_state` | `(B, conv_dim, conv_kernel_size - 1)` | 卷积的原始（未激活）输入历史 |
| `recurrent_state` | `(B, num_v_heads, head_k_dim, head_v_dim)` | delta rule 的矩阵状态，float32 |

约束与约定：

- `hidden_states` 为 `(B, N, hidden_size)`。
- `attention_mask` 是 **2D padding mask** `(B, N)`：bool 时 `True` 表示保留；
  float 可以是 0/1 乘性 mask，也接受 additive 约定（`0` 保留、负值屏蔽）。
  GDA 是递推层，不接受 `(B, 1, N, N)` 的逐 query attention mask；padding
  位置在投影前被置零。
- 省略 `attention_mask` 时不额外处理 padding。
- `use_cache=True` 时返回 `(output, state)`；否则只返回 `output`。完整序列
  前向与逐 token 解码在数值上一致（见测试）。
- 层返回的 `recurrent_state` 始终是 float32，避免长序列 fp16 溢出。
- 参数名与 Qwen3.5 checkpoint 对应：`in_proj_qkv`、`in_proj_z`、`in_proj_b`、
  `in_proj_a`、`conv1d.weight`、`dt_bias`、`A_log`、`norm.weight`、
  `out_proj.weight`。`Qwen3_5GatedDeltaNet` 是兼容别名。

## 5 实现

- 参考实现（逐 token oracle）：
  [attention/gda/reference.py](../attention/gda/reference.py)
- 纯 PyTorch 分块实现与解码递推：
  [attention/gda/torch.py](../attention/gda/torch.py)
- 参数化层与 gated RMSNorm：
  [attention/gda/layer.py](../attention/gda/layer.py)
- 示例：[examples/gda.py](../examples/gda.py)
- 测试：[tests/test_gda.py](../tests/test_gda.py)
- 论文：Gated Delta Networks: Improving Mamba2 with Delta Rule,
  <https://arxiv.org/abs/2412.06464>
- 参考实现：FLA `chunk_gated_delta_rule` / `fused_recurrent_gated_delta_rule`，
  <https://github.com/fla-org/flash-linear-attention>
- Qwen3.5：<https://github.com/huggingface/transformers>
