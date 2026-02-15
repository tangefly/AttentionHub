from attention import moba_attn_varlen_naive
import torch
import random

def generate_data(batch, seqlen, num_q_head, num_kv_head, headdim, dtype):
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    device = torch.cuda.current_device()

    # gen qkv
    q = torch.randn(
        (seqlen, num_q_head, headdim), dtype=dtype, device=device, requires_grad=True
    )
    k = torch.randn(
        (seqlen, num_kv_head, headdim), dtype=dtype, device=device, requires_grad=True
    )
    v = torch.randn(
        (seqlen, num_kv_head, headdim), dtype=dtype, device=device, requires_grad=True
    )

    # gen cu seqlen
    cu_seqlen = random.sample(range(1, seqlen - 1), batch - 1) if batch > 1 else []
    cu_seqlen.sort()
    cu_seqlen = [0] + cu_seqlen + [seqlen]
    cu_seqlen = torch.tensor(cu_seqlen, device=device, dtype=torch.int32)

    # max_seqlen
    max_seqlen = torch.amax(cu_seqlen[1:] - cu_seqlen[:-1])

    return q, k, v, cu_seqlen, max_seqlen.item()

def test_attn_varlen_moba(batch, head, seqlen, head_dim, moba_chunk_size, moba_topk):
    dtype = torch.bfloat16
    eps = 2e-2

    # Get data
    q, k, v, cu_seqlen, max_seqlen = generate_data(
        batch, seqlen, head, head, head_dim, dtype
    )
    vo_grad = torch.randn_like(q)

    # varlen func
    o_ref = moba_attn_varlen_naive(
        q,
        k,
        v,
        cu_seqlen,
        max_seqlen,
        moba_chunk_size=moba_chunk_size,
        moba_topk=moba_topk,
    )

batch, head, seqlen, head_dim, moba_chunk_size, moba_topk = 4, 4, 1024, 128, 256, 3
test_attn_varlen_moba(batch, head, seqlen, head_dim, moba_chunk_size, moba_topk)
