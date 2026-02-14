import torch
from torch import nn
import math

class PredictSparseAttention(nn.Module):
    def __init__(self, d, k, topk):
        super().__init__()

        self.topk = topk

        # Random Projection
        P = math.sqrt(3 / k) * torch.randint(-1, 2, (d, k)).float()
        self.register_buffer("P", P)

        # Approximation Path
        self.wq_tilde = nn.Linear(k, k, bias=False)
        self.wk_tilde = nn.Linear(k, k, bias=False)

        # Standard Attention
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)

    def forward(self, X):
        Xp = X @ self.P
        tQ = self.wq_tilde(Xp)
        tK = self.wk_tilde(Xp)

        tS = tQ @ tK.transpose(-1, -2)

        topk = torch.topk(tS, self.topk, dim=-1).indices
        mask = torch.ones_like(tS, dtype=torch.bool)
        mask.scatter_(-1, topk, False)

        Q = self.wq(X)
        K = self.wk(X)
        V = self.wv(X)

        S = Q @ K.transpose(-1, -2)
        S = S.masked_fill(mask, -1e9)

        A = torch.softmax(S, dim=-1)
        O = A @ V

        return O
