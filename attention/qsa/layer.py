from torch import nn
import torch
import math

from ..norm import QwenRMSNorm
from ..rope import apply_rotary_pos_emb

# Backwards-compatible name for the RMSNorm that used to live in this module.
RMSNorm = QwenRMSNorm


def repeat_kv(hidden_states, n_rep):
    """Repeat KV heads to match the number of query heads (GQA/MQA)."""
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0):
    """Plain PyTorch attention used after the QSA mask has been built."""
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            attn_weights = attn_weights.masked_fill(~attention_mask, torch.finfo(attn_weights.dtype).min)
        else:
            attn_weights = attn_weights + attention_mask

    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class QSAIndexer(nn.Module):
    """Select QSA token indices from compressed key blocks."""

    def __init__(
        self, 
        hidden_size,
        indexer_n_heads: int,
        indexer_kv_heads: int,
        indexer_head_dim: int,
        indexer_budget: int,
        indexer_compress_ratio: float,
        rms_norm_eps: float,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.index_n_heads = indexer_n_heads
        self.index_kv_heads = indexer_kv_heads
        self.index_head_dim = indexer_head_dim
        self.token_budget = indexer_budget
        self.compress_ratio = indexer_compress_ratio
        self.block_topk = self.token_budget // self.compress_ratio
        self.rms_norm_eps = rms_norm_eps
        
        self.index_qk_proj = nn.Linear(
            self.hidden_size,
            (self.index_n_heads + self.index_kv_heads) * self.index_head_dim,
            bias=False,
        )
        self.q_layernorm = QwenRMSNorm(self.index_head_dim, eps=self.rms_norm_eps)
        self.k_layernorm = QwenRMSNorm(self.index_head_dim, eps=self.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.shape
        hidden_shape = (batch_size, seq_length, -1, self.index_head_dim)
        full_cos, full_sin = position_embeddings
        current_cos, current_sin = full_cos[:, -seq_length:, :], full_sin[:, -seq_length:, :]

        qk = self.index_qk_proj(hidden_states)
        q, token_k = torch.split(
            qk,
            [self.index_n_heads * self.index_head_dim, self.index_kv_heads * self.index_head_dim],
            dim=-1,
        )
        q, raw_keys = q.reshape(*hidden_shape), token_k.reshape(*hidden_shape).squeeze(2)
        q = self.q_layernorm(q)
        q = apply_rotary_pos_emb(q, cos=current_cos, sin=current_sin, unsqueeze_dim=2)

        visible_token_indices = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0

        selected_token_indices = torch.full(
            (batch_size, seq_length, self.token_budget + self.compress_ratio - 1),
            -1,
            dtype=torch.int32,
            device=hidden_states.device,
        )
        for batch_idx in range(batch_size):
            for query_idx in range(seq_length):
                local_visible_indices = torch.nonzero(
                    visible_token_indices[batch_idx, 0, query_idx], as_tuple=False
                ).flatten()
                num_complete_blocks = local_visible_indices.shape[-1] // self.compress_ratio
                # Compute selected tokens
                if num_complete_blocks > 0:
                    block_token_indices = local_visible_indices[: num_complete_blocks * self.compress_ratio].view(
                        num_complete_blocks, self.compress_ratio
                    )

                    key_groups = raw_keys[batch_idx].index_select(0, block_token_indices.flatten())
                    key_groups = key_groups.view(*block_token_indices.shape, self.index_head_dim)
                    pooled_keys = key_groups.float().mean(dim=1).to(raw_keys.dtype)
                    pooled_keys = self.k_layernorm(pooled_keys)
                    group_starts = block_token_indices[:, 0]
                    block_key_states = apply_rotary_pos_emb(
                        pooled_keys.unsqueeze(1),
                        cos=full_cos[batch_idx].index_select(0, group_starts),
                        sin=full_sin[batch_idx].index_select(0, group_starts),
                    ).squeeze(1)

                    scores = torch.matmul(
                        q[batch_idx, query_idx].float(), block_key_states.float().transpose(-1, -2)
                    ).transpose(-1, -2)
                    scores = torch.relu(scores).sum(dim=-1) / math.sqrt(self.index_head_dim)

                    selected_block_indices = scores.topk(min(self.block_topk, num_complete_blocks), dim=0).indices
                    # Remap the indices of the blocks to the indices of individual tokens
                    selected_tokens = block_token_indices.index_select(0, selected_block_indices).flatten()
                else:
                    selected_tokens = torch.tensor([], device=hidden_states.device)
                tail = local_visible_indices[num_complete_blocks * self.compress_ratio :]
                selected_tokens = torch.cat([selected_tokens, tail]).to(torch.int32)
                selected_token_indices[batch_idx, query_idx, : selected_tokens.numel()] = selected_tokens

        kv_length = attention_mask.shape[-1]
        selected_token_mask = torch.zeros(
            (*selected_token_indices.shape[:-1], kv_length + 1), device=attention_mask.device, dtype=torch.bool
        )
        scatter_indices = torch.where(selected_token_indices >= 0, selected_token_indices, kv_length)
        selected_token_mask = selected_token_mask.scatter(-1, scatter_indices, True)[..., :kv_length].unsqueeze(1)
        if attention_mask.is_floating_point():
            min_dtype = torch.finfo(attention_mask.dtype).min
            selected_token_mask = torch.where(selected_token_mask, attention_mask.new_zeros(()), min_dtype)

        return selected_token_mask


class QwenSparseAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        head_dim: int,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        attention_dropout: float,
        attention_bias: bool,
        rms_norm_eps: float,
        indexer_n_heads: int,
        indexer_kv_heads: int,
        indexer_head_dim: int,
        indexer_budget: int,
        indexer_compress_ratio: float,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(
            hidden_size, num_attention_heads * self.head_dim * 2, bias=attention_bias
        )
        self.k_proj = nn.Linear(
            hidden_size, num_key_value_heads * self.head_dim, bias=attention_bias
        )
        self.v_proj = nn.Linear(
            hidden_size, num_key_value_heads * self.head_dim, bias=attention_bias
        )
        self.o_proj = nn.Linear(
            num_attention_heads * self.head_dim, hidden_size, bias=attention_bias
        )
        self.q_norm = QwenRMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = QwenRMSNorm(self.head_dim, eps=rms_norm_eps)
        self.indexer = QSAIndexer(
            hidden_size,
            indexer_n_heads,
            indexer_kv_heads,
            indexer_head_dim,
            indexer_budget,
            indexer_compress_ratio,
            rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        selected_token_mask = self.indexer(hidden_states, position_embeddings, attention_mask)
        # Combine both masks (they are never None, and are always 4D with either bool for sdpa, or float for eager)
        if attention_mask.is_floating_point():
            attention_mask = attention_mask + selected_token_mask
        else:
            attention_mask = attention_mask & selected_token_mask

        # The cos/sin are the full positions here due to the indexer, so we need to slice to get current positions
        position_embeddings = (x[:, -hidden_states.shape[1] :, :] for x in position_embeddings)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights