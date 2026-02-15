import math
import torch
from torch import Tensor
from typing import Optional, Tuple


def grouped_query_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Compute grouped-query attention, a generalization of multi-query attention which uses an intermediate (more than one, less than number of query heads) number of key-value heads.

    Args:
        Q: Query tensor of shape, [batch, q_heads, q_num, qk_dim]
        K: Key tensor of shape, [batch, kv_heads, kv_num, qk_dim]
        V: Value tensor of shape, [batch, kv_heads, kv_num, v_dim]
        mask: Optional attention mask. Positions with 0 will be masked.

    Returns:
        output: Attention output tensor
        attention_weights: Attention weight matrix
    """

    if Q.size(-1) != K.size(-1):
        raise ValueError("Q and K must have same feature dim")

    if K.size(-2) != V.size(-2):
        raise ValueError("K and V must have same seq len")

    batch, q_heads, q_num, qk_dim = Q.shape
    kv_heads = K.size(1)
    kv_num = K.size(2)

    if q_heads % kv_heads != 0:
        raise ValueError(f"hq ({q_heads}) must be divisible by hk ({kv_heads})")

    group_size = q_heads // kv_heads

    # reshape Q -> group, [batch, q_heads, q_num, qk_dim] -> [batch, q_heads, q_num, qk_dim]
    Q = Q.view(batch, kv_heads, group_size, q_num, qk_dim)

    # Step1: similarity
    scores = torch.einsum("bhgnd,bhmd->bhgnm", Q, K)

    scores = scores / math.sqrt(qk_dim)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # Step2: softmax
    attn = torch.softmax(scores, dim=-1)

    # Step3: value
    out = torch.einsum("bhgnm,bhmd->bhgnd", attn, V)

    # reshape back
    out = out.reshape(batch, q_heads, q_num, qk_dim)
    attn = attn.reshape(batch, q_heads, q_num, kv_num)

    return out, attn
