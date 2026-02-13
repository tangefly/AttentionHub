import math
import torch
from torch import Tensor
from typing import Optional, Tuple

def multi_query_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Compute Multi Query Attention. Multi-head attention consists of multiple attention layers (heads) in parallel with different linear transformations on the queries, keys, values and outputs. Multi-query attention is identical except that the different heads share a single set of keys and values.

    Args:
        Q: Query tensor of shape (batch, heads, n_queries, d_k) or (heads, n_queries, d_k)
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
    scores = torch.einsum("bhid,bjd->bhij", Q, K)

    # Step 2: scale
    scores = scores / math.sqrt(d_k)

    # Step 3: apply mask (if provided)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # Step 4: softmax normalization
    attention_weights = torch.softmax(scores, dim=-1)

    # Step 5: weighted sum of values
    output = torch.einsum("bhij,bjd->bhid", attention_weights, V)

    return output, attention_weights

