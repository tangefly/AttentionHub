import torch

from attention.gda import GatedDeltaNet, gated_delta_rule


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    torch.manual_seed(0)
    print("using device:", device)

    # ------------------------------------------------------------------
    # 1. The pure function: gated delta rule on (B, T, H, D) tensors.
    # ------------------------------------------------------------------
    batch, seq_len, heads = 2, 128, 4
    head_k_dim, head_v_dim = 32, 32
    q = torch.randn(batch, seq_len, heads, head_k_dim, device=device, dtype=dtype)
    k = torch.randn(batch, seq_len, heads, head_k_dim, device=device, dtype=dtype)
    v = torch.randn(batch, seq_len, heads, head_v_dim, device=device, dtype=dtype)
    g = -torch.rand(batch, seq_len, heads, device=device, dtype=dtype)
    beta = torch.rand(batch, seq_len, heads, device=device, dtype=dtype)

    out, final_state = gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        backend="torch",
    )
    print(f"gated_delta_rule output: {tuple(out.shape)}")
    print(f"recurrent state:        {tuple(final_state.shape)}")

    # ------------------------------------------------------------------
    # 2. The parameterized Qwen3.5-style GDA layer + incremental decoding.
    # ------------------------------------------------------------------
    hidden_size = 256
    layer = GatedDeltaNet(
        hidden_size=hidden_size,
        num_v_heads=6,
        num_k_heads=3,
        head_k_dim=32,
        head_v_dim=32,
        conv_kernel_size=4,
        rms_norm_eps=1e-6,
    ).to(device=device, dtype=dtype)
    layer.eval()

    hidden_states = torch.randn(
        batch, seq_len, hidden_size, device=device, dtype=dtype
    )
    with torch.no_grad():
        output, state = layer(hidden_states, use_cache=True)

        decoded = []
        step_state = None
        for t in range(seq_len):
            step, step_state = layer(
                hidden_states[:, t : t + 1], past_state=step_state, use_cache=True
            )
            decoded.append(step)
        decoded = torch.cat(decoded, dim=1)

    print(f"layer output:           {tuple(output.shape)}")
    print(f"conv state:             {tuple(state.conv_state.shape)}")
    print(f"recurrent state:        {tuple(state.recurrent_state.shape)}")
    print(
        "full vs token-by-token max error: "
        f"{(output - decoded).abs().max().item():.2e}"
    )
    print(
        "state memory (batch=2): "
        f"{state.recurrent_state.numel() * 4 / 1024:.1f} KiB"
    )


if __name__ == "__main__":
    main()
