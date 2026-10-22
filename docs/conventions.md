# 实现约定

为保证不同算法、不同后端可以互相替换和对比，AttentionHub 统一以下约定。

## 1. 函数与层级

- **纯函数算子**放在 `reference.py` / `torch.py` / `triton.py` / `cuda/` / `flash.py` 中，对外暴露同名函数，例如 `attention.sdpa.reference.sdpa` 和 `attention.sdpa.torch.sdpa`。
- **带参数的层**放在 `layer.py` 中，例如 `attention.psa.layer.PredictSparseAttention`、`attention.qsa.layer.QwenSparseAttention`、`attention.mla.layer.DeepseekV2Attention` 和 `attention.dsa.layer.DeepseekSparseAttention`。
- 每个算法目录的 `__init__.py` 暴露公共入口、`available_backends()` 和必要的兼容别名。

## 2. Tensor layout

| 场景 | Layout | 说明 |
| --- | --- | --- |
| Dense SDPA | `(..., N, D)` | 任意 leading dims，常用 `(B, H, N, D)` |
| MQA | Q `(B, H, N, D)`，K/V `(B, N, D)` | 也支持 Q `(H, N, D)`、K/V `(N, D)` |
| GQA | Q `(B, Hq, N, D)`，K/V `(B, Hkv, N, D)` | `Hq % Hkv == 0` |
| FTA | Q/K/V `(B, H, N, D)` | `D` 必须能被 2 整除 |
| PSA | `(B, N, D)` | 内部完成 Q/K/V 投影 |
| MoBA | Q/K/V `(T, H, D)` + `cu_seqlens` | packed variable-length，同 flash-attn |
| QSA | hidden `(B, N, D)` + `position_embeddings` + `attention_mask` | 内部完成 Q/K/V 投影与稀疏选择 |
| MLA | hidden `(B, N, D)` + `position_embeddings` + 可选 `attention_mask` | 内部完成低秩 Q/KV 投影；mask 省略时默认 causal |
| DSA | hidden `(B, N, D)` + `position_embeddings` + 可选 `attention_mask` | MLA + indexer top-k 稀疏选择；mask 省略时默认 causal |

## 3. Mask 语义

- **布尔 mask**：`True` 表示保留（attend），`False` 表示屏蔽。与 PyTorch 的 `scaled_dot_product_attention` 一致。
- **浮点 mask**：additive，`0` 表示保留，`-inf` 或 `finfo.min` 表示屏蔽。
- `causal=True` 与显式 `mask` 相互独立，可以同时使用；内部会把 causal mask 与用户 mask 合并。
- MoBA/QSA/DSA 的稀疏 mask 由算法内部生成，不需要用户传入。

## 4. 返回值

- 纯函数算子默认只返回 `output`。
- `return_attn=True` 时返回 `(output, attention_weights)`；只有 `reference` 后端保证支持，融合 kernel 可以明确拒绝。
- MoBA 和 FTA 目前不返回 attention weights。
- PSA/QSA/MLA/DSA 是层，返回 `(output, attention_weights)`。

## 5. Backend 选择

- 每个算法提供 `available_backends()`。
- 公共入口接受 `backend="auto"`，按 `cuda > triton > flash > torch > reference` 的顺序选择已安装且可用的后端。
- 指定不可用的 backend 会抛出 `ValueError`，并列出可用选项。
- 可选依赖必须懒加载：`import attention` 不能因为缺少 flash-attn / triton / CUDA 而失败。

## 6. 精度与测试

- 参考实现作为 oracle；每个 fast backend 都要和它做 allclose。
- 测试容差按 dtype：fp32 `2e-5`，fp16 `5e-3`，bf16 `2e-2`（可按算子调整）。
- 至少覆盖：非 2 的幂序列长度、GQA/MQA 分组、varlen 不均匀长度、bool/float mask、`causal`、fp16/bf16/fp32。
