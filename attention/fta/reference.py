import math
import torch
from torch import Tensor
from typing import Optional, Tuple

def build_stride_mask(
    num: int, 
    stride: int,
    device: torch.device
) -> Tensor:
    
    idx = torch.arange(num, device=device)

    i = idx[:, None]   # [num, 1]
    j = idx[None, :]   # [1, num]

    future = j > i
    not_stride = (i - j) % stride != 0

    mask = future | not_stride
    return mask   # True = mask

def factorized_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    l: int,
) -> Tuple[Tensor, Tensor]:
    """
    Compute Factorized Attention.

    Args:
        Q: Query tensor of shape, [batch, head, num, dim]
        K: Key tensor of shape, [batch, head, num, dim]
        V: Value tensor of shape, [batch, head, num, dim]
        mask: Optional attention mask. Positions with 0 will be masked.

    Returns:
        output: Attention output tensor
        attention_weights: Attention weight matrix
    """

    if Q.size(-1) != K.size(-1):
        raise ValueError(
            f"Q and K must have same feature dim, got {Q.size(-1)} and {K.size(-1)}"
        )

    if K.size(-2) != V.size(-2):
        raise ValueError(
            f"K and V must have same sequence length, got {K.size(-2)} and {V.size(-2)}"
        )
    
    fact_head = 2
    batch, head, num, dim = Q.shape
    Q = Q.view(batch, head, fact_head, num, dim // fact_head)
    K = K.view(batch, head, fact_head, num, dim // fact_head)
    V = V.view(batch, head, fact_head, num, dim // fact_head)

    idx = torch.arange(num, device=Q.device)
    local_mask = (idx[None, :] > idx[:, None]) | (idx[None, :] < (idx[:, None] - l))

    # 1. Compute LocalAttention
    Q1 = Q[:, :, 0, :, :]
    K1 = K[:, :, 0, :, :]
    V1 = V[:, :, 0, :, :]
    scores1 = torch.matmul(Q1, K1.transpose(-2, -1))
    scores1 = scores1 / math.sqrt(dim)
    scores1 = scores1.masked_fill(local_mask == True, float("-inf"))
    attention_weights1 = torch.softmax(scores1, dim=-1)
    output1 = torch.matmul(attention_weights1, V1)

    # 2. Compute Stride
    Q2 = Q[:, :, 1, :, :]
    K2 = K[:, :, 1, :, :]
    V2 = V[:, :, 1, :, :]
    stride_mask = build_stride_mask(num, l, Q.device)
    scores2 = torch.matmul(Q2, K2.transpose(-2, -1))
    scores2 = scores2 / math.sqrt(dim)
    scores2 = scores2.masked_fill(stride_mask == True, float("-inf"))
    attention_weights2 = torch.softmax(scores2, dim=-1)
    output2 = torch.matmul(attention_weights2, V2)

    # 3. Merge Result
    output = torch.stack([output1, output2], dim=-3)
    output = output.view(batch, head, num, dim)

    return output

