import math
import torch
from torch import Tensor
from typing import Optional, Tuple

def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Compute Scaled Dot-Product Attention.

    Args:
        Q: Query tensor of shape (batch, n_queries, d_k) or (n_queries, d_k)
        K: Key tensor of shape (batch, n_keys, d_k) or (n_keys, d_k)
        V: Value tensor of shape (batch, n_keys, d_v) or (n_keys, d_v)
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

    d_k = Q.size(-1)

    # Step 1: compute similarity scores QK^T
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # Step 2: scale
    scores = scores / math.sqrt(d_k)

    # Step 3: apply mask (if provided)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # Step 4: softmax normalization
    attention_weights = torch.softmax(scores, dim=-1)

    # Step 5: weighted sum of values
    output = torch.matmul(attention_weights, V)

    return output, attention_weights

