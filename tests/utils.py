from __future__ import annotations

import torch


def tolerance(dtype: torch.dtype) -> float:
    if dtype == torch.float32:
        return 2e-5
    if dtype == torch.bfloat16:
        return 2e-2
    return 5e-3
