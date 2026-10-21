# AttentionHub 文档索引

AttentionHub 收集注意力变体及其不同后端的实现。每个算法一个目录，后端由文件名表达：

```
attention/<algorithm>/
    reference.py   # 可读的参考实现（oracle）
    torch.py       # 纯 PyTorch / 内置算子实现
    triton.py      # Triton kernel
    cuda/          # 自研 CUDA extension
    flash.py       # flash-attn 等第三方后端
    layer.py       # 带参数的 nn.Module（如果该算法本身就是一层）
```

## 支持矩阵

| Year | Kernel | Docs | Paper | Reference | Torch | Flash | Triton | CUDA |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 2017 | Scaled Dot-Product Attention (SDPA) | [SDPA](sdpa.md) | [Link](https://arxiv.org/abs/1706.03762) | ✅ | ✅ | ❌ | ❌ | ❌ |
| 2019 | Factorized Attention (FTA) | [FTA](fta.md) | [Link](https://arxiv.org/abs/1904.10509) | ✅ | ❌ | ❌ | ❌ | ❌ |
| 2019 | Multi-Query Attention (MQA) | [MQA](mqa.md) | [Link](https://arxiv.org/abs/1911.02150) | ✅ | ❌ | ❌ | ❌ | ❌ |
| 2022 | Predictive Sparse Attention (PSA) | [PSA](psa.md) | [Link](https://arxiv.org/abs/2110.11299) | ✅ | ❌ | ❌ | ❌ | ❌ |
| 2023 | Grouped-Query Attention (GQA) | [GQA](gqa.md) | [Link](https://arxiv.org/abs/2305.13245) | ✅ | ❌ | ❌ | ❌ | ❌ |
| 2025 | Mixture of Block Attention (MoBA) | [MoBA](moba.md) | [Link](https://arxiv.org/abs/2502.13189) | ✅ | ❌ | ✅ | ❌ | ❌ |
| 2026 | Qwen Sparse Attention (QSA) | [QSA](qsa.md) | - | ✅ | ❌ | ❌ | ❌ | ❌ |

## 约定

所有实现共享的 tensor layout、mask 语义、返回值和 backend 选择规则见 [conventions.md](conventions.md)。
