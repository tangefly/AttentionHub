from __future__ import annotations

import pytest
import torch


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dtype(device: torch.device) -> torch.dtype:
    # CPU half-precision matmul is not consistently supported.
    return torch.float32 if device.type == "cpu" else torch.float16
