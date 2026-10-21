from attention.moba import moba
import torch


def main():
    seqlen, heads, head_dim = 64, 2, 16
    moba_chunk_size, moba_topk = 8, 4
    dtype = torch.float32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    torch.manual_seed(0)
    q = torch.randn(seqlen, heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(seqlen, heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(seqlen, heads, head_dim, device=device, dtype=dtype)

    cu_seqlens = torch.tensor([0, 24, seqlen], device=device, dtype=torch.int32)
    max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())

    output = moba(
        q,
        k,
        v,
        cu_seqlens,
        max_seqlen,
        moba_chunk_size=moba_chunk_size,
        moba_topk=moba_topk,
    )

    print(f"output shape: {output.shape}")
    print("backend:", "flash" if device.type == "cuda" else "reference")


if __name__ == "__main__":
    main()
