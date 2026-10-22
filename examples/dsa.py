import torch

from attention.dsa import DeepseekSparseAttention
from attention.rope import build_rope_cache_batched


def main():
    batch, seq_len = 32, 512
    hidden_size = 7168
    num_heads = 128
    q_lora_rank, kv_lora_rank = 1536, 512
    qk_nope_head_dim, qk_rope_head_dim = 128, 64
    v_head_dim = 128
    index_n_heads, index_head_dim, index_topk = 64, 128, 2048

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    print("using device:", device)

    layer = DeepseekSparseAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        q_lora_rank=q_lora_rank,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        index_rope_head_dim=qk_rope_head_dim,
        index_topk=index_topk,
        attention_dropout=0.0,
        attention_bias=False,
        rms_norm_eps=1e-6,
        rope_layout="interleaved",
    ).to(device=device, dtype=dtype)
    layer.eval()

    hidden_states = torch.randn(
        batch, seq_len, hidden_size, device=device, dtype=dtype
    )
    cos, sin = build_rope_cache_batched(
        batch, seq_len, qk_rope_head_dim, device=device, dtype=dtype
    )

    # Additive causal mask: (batch, 1, seq_len, seq_len).
    causal = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).tril()
    attention_mask = torch.zeros(
        batch, 1, seq_len, seq_len, device=device, dtype=dtype
    ).masked_fill(~causal, torch.finfo(dtype).min)

    with torch.no_grad():
        output, attention_weights = layer(
            hidden_states, (cos, sin), attention_mask
        )
        q_resid = layer.q_a_layernorm(layer.q_a_proj(hidden_states))
        indices = layer.indexer(
            hidden_states, q_resid, (cos, sin), attention_mask[:, 0]
        )

    kept = (attention_weights > 0).sum(dim=-1).float()
    print(f"output shape: {output.shape}")
    print(f"attention_weights shape: {attention_weights.shape}")
    print(f"top-k indices shape: {indices.shape}")
    print(f"index_topk: {index_topk}")
    print(f"mean kept keys per query: {kept.mean().item():.1f}")
    print(
        "sparsity vs dense: "
        f"{1.0 - kept.mean().item() / seq_len:.1%}"
    )


if __name__ == "__main__":
    main()
