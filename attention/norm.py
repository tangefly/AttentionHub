"""RMSNorm variants shared by the layer implementations.

Two parameterizations are used in the hub:

* :class:`RMSNorm` keeps a learnable scale initialized to one, so the layer
  computes ``weight * x / rms(x)``.  This is the norm used by DeepSeek MLA and
  by the DSA main attention.
* :class:`QwenRMSNorm` stores a zero-initialized offset and computes
  ``(1 + weight) * x / rms(x)``.  This is the norm used by the QSA indexer and
  by the QSA query/key normalization.

Both variants normalize in float32 and cast the result back to the input
dtype, matching the reference implementations they replace.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["QwenRMSNorm", "RMSNorm"]


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization.

    Args:
        dim: Size of the normalized (last) dimension.
        eps: Numerical stability constant added to the mean square.
        group_size: If given, the last dimension is split into groups of this
            size and RMS normalization is applied inside each group.  The
            default ``None`` normalizes over the whole last dimension.
        unit_offset: If ``True``, the learnable parameter is zero-initialized
            and interpreted as an offset: the effective scale is
            ``1 + weight``.  The default ``False`` uses ``weight`` directly and
            initializes it to one.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        *,
        group_size: int | None = None,
        unit_offset: bool = False,
    ) -> None:
        super().__init__()
        if group_size is not None and dim % group_size != 0:
            raise ValueError(
                f"hidden_size ({dim}) must be divisible by group_size ({group_size})."
            )
        self.eps = eps
        self.group_size = group_size
        self.unit_offset = unit_offset
        self.weight = nn.Parameter(
            torch.zeros(dim) if unit_offset else torch.ones(dim)
        )

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        if self.group_size is not None:
            x = x.reshape(*x.shape[:-1], -1, self.group_size)
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return out.flatten(-2) if self.group_size is not None else out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float())
        weight = (1.0 + self.weight) if self.unit_offset else self.weight
        output = output * weight.float()
        return output.type_as(x)

    def extra_repr(self) -> str:
        parts = [f"{tuple(self.weight.shape)}", f"eps={self.eps}"]
        if self.group_size is not None:
            parts.append(f"group_size={self.group_size}")
        if self.unit_offset:
            parts.append("unit_offset=True")
        return ", ".join(parts)


class QwenRMSNorm(RMSNorm):
    """Qwen-style RMSNorm used by QSA.

    Equivalent to ``RMSNorm(..., unit_offset=True)`` but with the argument
    order of the RMSNorm that used to live in :mod:`attention.qsa.layer`.
    """

    def __init__(
        self,
        dim: int,
        group_size: int | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(
            dim,
            eps=eps,
            group_size=group_size,
            unit_offset=True,
        )
