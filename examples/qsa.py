from attention.qsa import QwenSparseAttention
import torch


def main():
    b, n = 2, 128
    hidden_size = 256
    num_heads, num_kv_heads, head_dim = 4, 2, 64
    indexer_n_heads, indexer_kv_heads, indexer_head_dim = 2, 1, 32
    indexer_budget, indexer_compress_ratio = 32, 8
    rotary_dim = 32
    dtype = torch.float32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    qsa = QwenSparseAttention(
        head_dim=head_dim,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        attention_dropout=0.0,
        attention_bias=False,
        rms_norm_eps=1e-6,
        indexer_n_heads=indexer_n_heads,
        indexer_kv_heads=indexer_kv_heads,
        indexer_head_dim=indexer_head_dim,
        indexer_budget=indexer_budget,
        indexer_compress_ratio=indexer_compress_ratio,
    ).to(device)
    qsa.eval()

    hidden_states = torch.randn(b, n, hidden_size, device=device, dtype=dtype)

    # RoPE cos/sin for every position: (batch, seq_len, rotary_dim)
    angles = (
        torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32)
        / rotary_dim
    )
    inv_freq = 1.0 / (10000**angles)
    freqs = torch.outer(torch.arange(n, device=device, dtype=torch.float32), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().unsqueeze(0).expand(b, -1, -1)
    sin = emb.sin().unsqueeze(0).expand(b, -1, -1)

    # Additive causal mask: (batch, 1, seq_len, seq_len)
    causal = torch.ones(n, n, device=device, dtype=torch.bool).tril()
    attention_mask = torch.zeros(b, 1, n, n, device=device, dtype=dtype)
    attention_mask = attention_mask.masked_fill(~causal, torch.finfo(dtype).min)

    with torch.no_grad():
        output, attention_weights = qsa(hidden_states, (cos, sin), attention_mask)

        # The indexer builds the sparse token mask applied on top of the causal mask
        selected_token_mask = qsa.indexer(hidden_states, (cos, sin), attention_mask)

    print(f"output shape: {output.shape}")
    print(f"attention_weights shape: {attention_weights.shape}")
    print(f"selected_token_mask shape: {selected_token_mask.shape}")
    if selected_token_mask.is_floating_point():
        kept = (selected_token_mask[0, 0] == 0).sum(dim=-1).float()
    else:
        kept = selected_token_mask[0, 0].sum(dim=-1).float()
    print(
        f"average kept tokens per query: {kept.mean().item():.1f} "
        f"(budget={indexer_budget})"
    )


if __name__ == "__main__":
    main()
