## 1 Install

```
git clone https://github.com/tangefly/AttentionHub.git
cd AttentionHub
pip install -e .
```

## 2 Support Kernels

|            Year            |            Kernel            |                           Docs                           |                           Source                           |                           Reference                           |                           Torch                           |                           Triton                           |                           Cuda                           |
| :--------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2017 | Scaled Dot Product Attention (SDPA) | [SDPA.md](https://github.com/tangefly/AttentionHub/blob/main/docs/SDPA.md) | [Link](https://arxiv.org/abs/1706.03762) | ✅ | ❌ | ❌ | ❌ |
| 2019 | Multi Auery Attention (MQA) | [MQA.md](https://github.com/tangefly/AttentionHub/blob/main/docs/MQA.md) | [Link](https://arxiv.org/abs/1911.02150) | ✅ | ❌ | ❌ | ❌ |
| 2019 | Factorized Attention (FTA) | [FTA.md](https://github.com/tangefly/AttentionHub/blob/main/docs/FTA.md) | [Link](https://arxiv.org/abs/1904.10509) | ✅ | ❌ | ❌ | ❌ |
| 2022 | Predict Sparse Attention (PSA) | [PSA.md](https://github.com/tangefly/AttentionHub/blob/main/docs/PSA.md) | [Link](https://arxiv.org/abs/2110.11299) | ✅ | ❌ | ❌ | ❌ |
| 2023 | Grouped Query Attention (GQA) | [GQA.md](https://github.com/tangefly/AttentionHub/blob/main/docs/GQA.md) | [Link](https://arxiv.org/abs/2305.13245) | ✅ | ❌ | ❌ | ❌ |
| 2025 | Mixture of Block Attention (MoBA) | [MoBA.md](https://github.com/tangefly/AttentionHub/blob/main/docs/MoBA.md) | [Link](https://arxiv.org/abs/2502.13189) | ✅ | ❌ | ❌ | ❌ |