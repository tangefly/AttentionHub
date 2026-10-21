import torch

from attention.mla import MultiHeadLatentAttention
from attention.rope import build_rope_cache_batched


def main():
    batch, seq_len = 2, 64
    hidden_size = 256
    num_heads = 8
    kv_lora_rank = 32
    qk_nope_head_dim, qk_rope_head_dim = 32, 16
    v_head_dim = 32
    q_lora_rank = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    print("using device:", device)

    layer = MultiHeadLatentAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        q_lora_rank=q_lora_rank,
        attention_dropout=0.0,
        attention_bias=False,
        rms_norm_eps=1e-6,
        rope_layout="interleaved",
    ).to(device=device, dtype=dtype)
    layer.eval()

    hidden_states = torch.randn(batch, seq_len, hidden_size, device=device, dtype=dtype)
    
    cos, sin = build_rope_cache_batched(
        batch,
        seq_len,
        qk_rope_head_dim,
        device=device,
        dtype=dtype,
    )

    with torch.no_grad():
        output, attention_weights = layer(hidden_states, (cos, sin))

    print(f"output shape: {output.shape}")
    print(f"attention_weights shape: {attention_weights.shape}")

    cache_per_token = kv_lora_rank + qk_rope_head_dim
    mha_cache_per_token = 2 * num_heads * (qk_nope_head_dim + qk_rope_head_dim)
    print(f"MLA cache per token: {cache_per_token} elements")
    print(f"MHA-like cache per token: {mha_cache_per_token} elements")
    print(f"compression ratio: {mha_cache_per_token / cache_per_token:.2f}x")


if __name__ == "__main__":
    main()
