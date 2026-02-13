import torch.nn.functional as F
from attention import scaled_dot_product_attention
from benchmark_attention import *

def my_attn(Q, K, V):
    return scaled_dot_product_attention(Q, K, V)[0]


def torch_attn(Q, K, V):
    return F.scaled_dot_product_attention(Q, K, V)


ops = {
    "mine": my_attn,
    "torch": torch_attn,
}

sizes = [512, 1024, 2048, 4096]

perf = sweep_seq_len(sizes, ops)

plot_attention_benchmark(
    sizes=sizes,
    perf_data=perf,
    out_name="sdpa_vs_torch.png",
    title="Scaled Dot Product Attention",
)
