# RMSNorm

MLA 和 QSA 都需要 RMSNorm，但两者的参数化方式不同：

| 变体 | 可学习参数初始化 | 前向缩放 | 使用位置 |
| --- | --- | --- | --- |
| `RMSNorm` | 全 1 | `weight * x / rms(x)` | MLA 的 query latent / KV latent |
| `QwenRMSNorm` | 全 0 | `(1 + weight) * x / rms(x)` | QSA 的 q/k norm 与 indexer |

两者都在 [attention/norm.py](../attention/norm.py) 中实现，避免各层重复维护同一段归一化逻辑。

## 1 数学形式

对最后一维（或每个 group）计算：

```
rms(x) = sqrt(mean(x^2) + eps)
y = scale * x / rms(x)
```

其中 `scale` 为：

- `RMSNorm`：`weight`，初始化为 1。
- `QwenRMSNorm`：`1 + weight`，`weight` 初始化为 0。零初始化让新加的 norm
  在训练开始时等价于恒等缩放。

归一化内部始终使用 float32 计算，输出再转回输入 dtype，和 fp16/bf16 训练时的
数值行为一致。

## 2 接口

```python
from attention.norm import QwenRMSNorm, RMSNorm

# 标准 RMSNorm（MLA）
norm = RMSNorm(128, eps=1e-6)

# Qwen / QSA 风格，权重为 1 + weight
qwen_norm = QwenRMSNorm(128, eps=1e-6)

# 可选：在最后一维内部按 group 归一化
grouped = QwenRMSNorm(128, group_size=32, eps=1e-6)
```

`RMSNorm` 的完整签名：

```python
RMSNorm(
    dim: int,
    eps: float = 1e-6,
    *,
    group_size: int | None = None,
    unit_offset: bool = False,
)
```

`QwenRMSNorm(dim, group_size=None, eps=1e-6)` 是
`RMSNorm(..., unit_offset=True)` 的便捷封装，保留了 QSA 原来的参数顺序。

约束：`group_size` 必须整除 `dim`。

## 3 在各层中的使用

MLA：

```python
from attention.mla import DeepseekV2Attention

layer = DeepseekV2Attention(...)
# layer.q_a_layernorm / layer.kv_a_layernorm 都是 RMSNorm
```

QSA：

```python
from attention.qsa import QwenSparseAttention

layer = QwenSparseAttention(...)
# layer.q_norm / layer.k_norm / layer.indexer.q_layernorm
# / layer.indexer.k_layernorm 都是 QwenRMSNorm
```

## 4 测试

- [tests/test_norm.py](../tests/test_norm.py)：两种参数化、group 归一化、
  dtype 转换和异常输入。
- [tests/test_qsa.py](../tests/test_qsa.py)：QSA 改用共享 RMSNorm 后仍与 dense
  attention 对齐。
- [tests/test_mla.py](../tests/test_mla.py)：MLA 使用共享 RMSNorm 后的数值正确性。
